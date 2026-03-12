"""
Train Safe RL agents on Safety-Gymnasium tasks.

This script glues together PPO, SAC, LagPPO, and a fixed-penalty RCPO baseline
from Stable-Baselines3 with optional geometric shielding.

The core question being studied is how different safety enforcement strategies
(Lagrangian constraints, fixed penalties, runtime shields) interact under realistic
compute constraints on Safety-Gymnasium tasks.

RCPO here is a simple fixed-penalty baseline: r' = r - penalty_coef * c.
It is not the full multi-timescale algorithm from Tessler et al. (2018),
but it serves as a useful static comparison point against adaptive methods like LagPPO.
"""

import os
import argparse
from typing import Any, Callable, Optional
import numpy as np
import torch
import multiprocessing as mp
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .utils.logging import Logger
from .envs.make_env import make_env
from .algos.ppo_sb3 import make_ppo
from .algos.sac_sb3 import make_sac
from .algos.lagppo import make_lagppo, LagrangianState
from .algos.rcpo import make_rcpo  # kept for future integration of a full RCPO pipeline
from .utils.serialization import save_config
from .utils.seeding import seed_everything
from .callbacks.train_logging_callback import TrainLoggingCallback
from .safety.shield import GenericKeepoutShield
from .reward.preferences.preference_dataset import TrajectoryBuffer
from .reward.preferences.preference_model import PreferenceReward, flatten_obs, train_preference_model
from .reward.preferences.wrappers import RewardReplacementWrapper
from gymnasium import RewardWrapper  # Used if algo == "rcpo"


def parse_args() -> argparse.Namespace:
    """
    Parse CLI flags for training.

    We expose the main knobs as command-line flags so experiments are fully
    described by the command: env, algorithm, safety hyperparameters, logging paths.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(
        description="Train a (safe) RL agent with constrained exploration."
    )
    ap.add_argument("--env_id", type=str, required=True,
                    help="Gymnasium/Safety-Gymnasium environment ID.")
    ap.add_argument(
        "--algo",
        type=str,
        default="lagppo",
        choices=["ppo", "sac", "rcpo", "lagppo"],
        help="Which algorithm to train.",
    )
    ap.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total number of environment timesteps for training.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for training and environment creation.",
    )
    ap.add_argument(
        "--cost_budget",
        type=float,
        default=0.05,
        help="Average per-step cost budget for Lagrangian methods.",
    )
    ap.add_argument(
        "--lr_lambda",
        type=float,
        default=5e-4,
        help="Learning rate for the Lagrange multiplier in LagPPO.",
    )
    ap.add_argument(
        "--use_shield",
        action="store_true",
        help="Enable geometric action shielding in the environment.",
    )
    ap.add_argument(
        "--use_preferences",
        action="store_true",
        help="Use a learned preference-based reward model instead of env reward.",
    )
    ap.add_argument(
        "--pref_steps",
        type=int,
        default=20000,
        help="Number of random steps to collect for preference dataset.",
    )
    ap.add_argument(
        "--log_dir",
        type=str,
        default="logs/quickstart_pointpush",
        help="Directory for JSON log files.",
    )
    ap.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Evaluation frequency (in timesteps) for callbacks.",
    )
    ap.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments (1 = DummyVecEnv, >1 = SubprocVecEnv).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device for SB3 (e.g. 'cpu', 'cuda', or 'auto').",
    )
    ap.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (handled inside Logger if configured).",
    )
    ap.add_argument(
        "--penalty_coef",
        type=float,
        default=1.0,
        help="RCPO fixed penalty coefficient for cost shaping.",
    )
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Override checkpoint directory. Defaults to checkpoints/<env_id>/<algo>/seed_<seed>.",
    )
    return ap.parse_args()


def build_shield_factory(env_id: str) -> Callable[[Any], GenericKeepoutShield]:
    """
    Build a factory that constructs a geometric keep-out shield for a given env.

    The returned factory is what matters at runtime: it takes a concrete environment
    instance, introspects its hazard layout when available, and returns a configured
    GenericKeepoutShield. The env_id argument is not used in the current logic but
    is kept so future versions can specialize shield parameters per task
    (e.g. different dt or hazard radii for different environments).

    If hazard introspection fails, the shield is left with an empty hazard list
    and becomes a no-op rather than crashing.

    :param env_id: Environment ID.
        :type env_id: str

    :return: A callable that maps an environment instance to a configured shield.
        :rtype: Callable[[Any], GenericKeepoutShield]
    """
    if env_id.startswith("mujoco:"):
        from mujoco_connector import MujocoShield
        return lambda env: MujocoShield()

    def factory(env: Any) -> GenericKeepoutShield:
        # Start with an empty hazard list; we fill it via introspection below
        shield = GenericKeepoutShield(
            hazards=[],
            dt=0.1,
            max_action_norm=float(np.max(env.action_space.high)),
        )

        # Safety-Gymnasium exposes hazard geometry through different attribute paths
        # depending on the version. We try each path in order of preference.
        try:
            uw = env.unwrapped

            hazards_pos = None
            hazards_size = 0.2  # default radius for point tasks in Safety-Gymnasium

            if hasattr(uw, "task") and hasattr(uw.task, "hazards"):
                h = uw.task.hazards
                hazards_pos = getattr(h, "pos", None)
                hazards_size = float(getattr(h, "size", 0.2))
            elif hasattr(uw, "task") and hasattr(uw.task, "hazards_pos"):
                # Older Safety-Gymnasium layout where pos lives directly on task
                hazards_pos = uw.task.hazards_pos
                hazards_size = float(getattr(uw.task, "hazards_size", 0.2))
            elif hasattr(uw, "world") and hasattr(uw.world, "hazards_pos"):
                # Legacy Safety-Gym (pre-0.4) attribute path
                hazards_pos = uw.world.hazards_pos
                hazards_size = float(getattr(uw.world, "hazards_size", 0.2))

            if hazards_pos is not None and len(hazards_pos) > 0:
                # Hazard positions are 3D (x, y, z); the shield only needs x, y
                hz = [(float(p[0]), float(p[1]), hazards_size) for p in hazards_pos]
                shield.set_hazards(hz)
                print(f"[Shield] loaded {len(hz)} hazards: {hz}")
            else:
                print("[Shield] WARNING: could not find hazard positions, shield is pass-through")

        except Exception as e:
            print(f"[Shield] WARNING: hazard introspection failed: {e}")

        return shield

    return factory


def collect_random_rollouts(env_id: str, steps: int, seed: int) -> TrajectoryBuffer:
    """
    Collect random trajectories for building the preference dataset.

    We run a random policy and store (obs, action, reward) triples in a
    TrajectoryBuffer. These are later used to construct synthetic preference
    pairs for training the reward model.

    :param env_id: ID of the environment to sample from.
        :type env_id: str
    :param steps: Total number of environment steps to collect.
        :type steps: int
    :param seed: Random seed for the environment and buffer.
        :type seed: int

    :return: A trajectory buffer containing the collected rollouts.
        :rtype: TrajectoryBuffer
    """
    env = make_env(env_id, seed=seed)
    buf = TrajectoryBuffer(segment_len=25, seed=seed)
    traj = []
    obs, info = env.reset()

    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        traj.append((obs, action, float(reward)))
        obs = next_obs
        if terminated or truncated:
            buf.add_trajectory(traj)
            traj = []
            obs, info = env.reset()

    if traj:
        buf.add_trajectory(traj)

    return buf


def main():
    """
    Orchestrate training for a single algorithm and shield condition.

    Steps:
      1. Parse CLI arguments and set global seeds
      2. Optionally pre-train a preference-based reward model from random rollouts
      3. Create a shared Lagrange multiplier (lambda) for LagPPO workers
      4. Build a vectorized environment with optional shield and reward shaping
      5. Instantiate the chosen algorithm (PPO, SAC, LagPPO, RCPO baseline)
      6. Train for the requested number of timesteps with logging callbacks
      7. Save the final checkpoint and config for downstream evaluation
    """
    args = parse_args()
    seed_everything(args.seed)

    # Logger setup
    os.makedirs(args.log_dir, exist_ok=True)
    tb_dir = os.path.join("runs", os.path.basename(args.log_dir))
    logger = Logger(log_dir=args.log_dir, tb_dir=tb_dir)
    tensorboard_log = logger.tb_dir

    # Preference model pre-training (offline, before the main training loop)
    pref_model: Optional[PreferenceReward] = None
    if args.use_preferences:
        print("Collecting random rollouts for preference dataset.")
        buf = collect_random_rollouts(
            args.env_id,
            steps=args.pref_steps,
            seed=args.seed,
        )
        pairs = buf.sample_segments(
            n_pairs=max(1, args.pref_steps // 100),
            noise=0.1,
        )

        # Infer observation dimensionality from the first collected sample
        sample_obs: Optional[Any] = None
        for traj in buf.trajectories:
            if traj:
                sample_obs = traj[0][0]
                break

        if sample_obs is None:
            raise RuntimeError(
                "Failed to collect any observations for preference model."
            )

        obs_dim = flatten_obs(sample_obs).shape[0]
        pref_model = PreferenceReward(input_dim=obs_dim, hidden_sizes=[128, 128])
        print("Training preference model.")
        train_preference_model(
            pref_model,
            pairs,
            epochs=5,
            batch_size=32,
            lr=1e-3,
            device="cpu",
        )

    # Shared lambda is updated by the Lagrangian callback and read by SubprocVecEnv workers.
    # We use a multiprocessing Manager so the value is safely shared across processes.
    manager = mp.Manager()
    shared_lambda = manager.Value("d", 0.0)

    def get_lambda_shared() -> float:
        """
        Read the current shared Lagrange multiplier.

        Passed to make_env so reward shaping inside worker processes always
        sees the most recent lambda value from the callback.

        :return: Current lambda value.
            :rtype: float
        """
        return shared_lambda.value

    def make_thunk(rank: int) -> Callable[[], Any]:
        """
        Build an initializer for a single environment instance.

        Each worker gets a unique seed offset so environments are independent.
        The initializer optionally adds a shield, RCPO reward shaping,
        and a learned preference reward model.

        :param rank: Worker index, used to offset the seed.
            :type rank: int

        :return: A zero-argument callable that creates and wraps the environment.
            :rtype: Callable[[], Any]
        """

        def _init():
            env = make_env(
                args.env_id,
                seed=args.seed + 1000 * rank,
                use_shield=args.use_shield,
                shield_factory=(
                    build_shield_factory(args.env_id)
                    if args.use_shield
                    else None
                ),
                # LagPPO and RCPO both need the current lambda for reward shaping
                reward_shaping_get_lambda=(
                    get_lambda_shared
                    if args.algo in ["lagppo", "rcpo"]
                    else None
                ),
            )

            if args.algo == "rcpo":
                # RCPO here is a fixed-penalty baseline: r' = r - penalty_coef * c.
                # The penalty does not adapt during training, which makes it stable
                # but less responsive to constraint violations than LagPPO.
                # A more faithful RCPO pipeline can reuse make_rcpo from src/algos/rcpo.py.

                class FixedPenalty(RewardWrapper):
                    """
                    Reward wrapper implementing r' = r - penalty_coef * cost.

                    The penalty coefficient is fixed for the entire training run.
                    This is intentionally simpler than the original RCPO algorithm,
                    serving as a static comparison point against adaptive methods.

                    :param env: Environment to wrap.
                        :type env: Any
                    :param penalty: Fixed penalty coefficient.
                        :type penalty: float
                    """

                    def __init__(self, env: Any, penalty: float) -> None:
                        super().__init__(env)
                        self.penalty: float = penalty

                    def step(self, action: Any):
                        """
                        Step the environment and apply the fixed cost penalty.

                        :param action: Action from the policy.
                            :type action: Any

                        :return: (obs, shaped_reward, terminated, truncated, info).
                            :rtype: tuple
                        """
                        obs, r, term, trunc, info = self.env.step(action)
                        cost = float(info.get("cost", 0.0))
                        shaped = float(r) - self.penalty * cost
                        info["shaped_reward"] = shaped
                        info["lambda"] = self.penalty
                        info["cost"] = cost
                        return obs, shaped, term, trunc, info

                env = FixedPenalty(env, penalty=args.penalty_coef)

            if args.use_preferences and pref_model is not None:
                # Replace the environment reward with the learned preference reward
                env = RewardReplacementWrapper(env, pref_model, device="cpu")

            return env

        return _init

    # Use SubprocVecEnv when running multiple workers for speed;
    # DummyVecEnv is simpler and easier to debug for single-env runs
    vec_env_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    vec_env = vec_env_cls([make_thunk(i) for i in range(args.num_envs)])

    # Algorithm selection
    lag_callback = None
    if args.algo == "ppo":
        model = make_ppo(
            policy="MlpPolicy",
            env=vec_env,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            config=None,
        )
        lag_state = None

    elif args.algo == "sac":
        model = make_sac(
            policy="MlpPolicy",
            env=vec_env,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            config=None,
        )
        lag_state = None

    elif args.algo == "lagppo":
        lag_state = LagrangianState(
            lam=0.0,
            lr_lambda=args.lr_lambda,
            cost_budget=args.cost_budget,
            update_every=2048,
        )
        model, lag_callback = make_lagppo(
            policy="MlpPolicy",
            env=vec_env,
            lag_state=lag_state,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            config=None,
        )
        # Give the callback access to shared_lambda so it can synchronize
        # the multiplier value across SubprocVecEnv worker processes
        lag_callback.shared_lambda_value = shared_lambda
        lag_callback.custom_logger = logger

    elif args.algo == "rcpo":
        # RCPO uses the FixedPenalty wrapper above for reward shaping
        # and a standard PPO backbone for optimization
        model = make_ppo(
            policy="MlpPolicy",
            env=vec_env,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            config=None,
        )
        lag_state = None

    else:
        raise ValueError(f"Unsupported algo: {args.algo}")

    # Training
    train_cb = TrainLoggingCallback(
        custom_logger=logger,
        log_freq=args.eval_freq,
        cost_budget=args.cost_budget
    )

    callbacks = [train_cb]
    if lag_callback is not None:
        callbacks.append(lag_callback)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    # Save the final checkpoint so evaluate.py can load it with --model_path .../latest
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else \
        os.path.join("checkpoints", args.env_id, args.algo, f"seed_{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest")
    model.save(latest_path)
    print(f"[train] Training done. Model saved to {latest_path}.zip")

    # Save config alongside the checkpoint for reproducibility
    cfg = vars(args).copy()
    cfg["torch_version"] = torch.__version__
    cfg["numpy_version"] = np.__version__
    save_config(cfg, ckpt_dir)

    vec_env.close()
    logger.close()


if __name__ == "__main__":
    main()
