"""
Train Safe RL agents on Safety-Gymnasium style tasks.

This script glues together:
- PPO / SAC backbones from Stable-Baselines3
- a Lagrangian PPO (LagPPO) variant with a cost budget
- a simple fixed-penalty RCPO-style baseline
- optional geometric shielding and preference-based rewards

The core idea is to study how different ways of enforcing safety behave under compute constraints on Safety-Gymnasium tasks.
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
    CLI flags for training (env, algo, safety knobs, logging, etc.).

    :return: Parsed command-line arguments.
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
    return ap.parse_args()


def build_shield_factory(env_id: str) -> Callable[[Any], GenericKeepoutShield]:
    """
    Build a factory that constructs a geometric keep-out shield for a given env.

    The factory tries to introspect Safety-Gymnasium environments to find hazard
    positions (e.g. env.unwrapped.world.hazards_pos) and uses them to build
    a GenericKeepoutShield. If introspection fails, the shield is left
    with an empty hazard list, effectively becoming a no-op.
	
	Personal note for a future update: the returned factory is actually what matters at runtime. It takes a concrete
    environment instance, introspects its hazard layout (when available), and returns a configured GenericKeepoutShield. 
	The env_id argument is currently not used in the logic, but it is kept so that future versions can specialize
    the shield parameters for different tasks (e.g. different dt or hazard radii).

    :param env_id: Environment ID (currently unused, but kept for future
        environment-specific shield configuration).
    :type env_id: str

    :return: A callable that maps an environment instance to a configured shield.
    :rtype: Callable[[Any], GenericKeepoutShield]
    """

    def factory(env: Any) -> GenericKeepoutShield:
        # Initialize with empty hazards; we may fill them via env introspection.
        shield = GenericKeepoutShield(
            hazards=[],
            dt=0.1,
            max_action_norm=float(np.max(env.action_space.high)),
        )
        # Best-effort attempt to read hazard positions from Safety-Gymnasium internals.
        try:
            uw = env.unwrapped
            if hasattr(uw, "world") and hasattr(uw.world, "hazards_pos"):
                positions = uw.world.hazards_pos  # list of 2D centers
                radius = getattr(uw.world, "hazards_size", 0.5)
                hz = [(float(x), float(y), float(radius)) for (x, y) in positions]
                shield.set_hazards(hz)
        except Exception:
            # If the environment structure changes, we simply fall back to an
            # empty shield rather than failing hard.
            pass

        return shield

    return factory


def collect_random_rollouts(env_id: str, steps: int, seed: int) -> TrajectoryBuffer:
    """
    Collect random trajectories from the environment for preference learning.

    This function runs a random policy in the given environment and stores
    trajectories of (observation, action, reward) triples in a
    TrajectoryBuffer. These trajectories are later used to construct
    synthetic preference pairs for training a reward model.

    :param env_id: ID of the environment to sample from.
    :type env_id: str
    :param steps: Total number of environment steps to collect.
    :type steps: int
    :param seed: Random seed for the environment and buffer.
    :type seed: int

    :return: A trajectory buffer containing the collected random rollouts.
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
    Main steps:
      1. Parse CLI arguments and set global seeds
      2. Optionally pre-train a preference-based reward model from random rollouts
      3. Create a shared Lagrange multiplier (lambda) for LagPPO / RCPO shaping
      4. Build a vectorized environment with optional shield and reward shaping
      5. Instantiate the chosen algorithm (PPO, SAC, LagPPO, RCPO baseline)
      6. Train for the requested number of timesteps with logging callbacks

    The function orchestrates all components without embedding algorithm-specific
    logic, so that the same training loop can be used for different safe RL
    configurations.
    """
	
    args = parse_args()
    seed_everything(args.seed)

    # Logger setup
    os.makedirs(args.log_dir, exist_ok=True)
    tb_dir = os.path.join("runs", os.path.basename(args.log_dir))
    logger = Logger(log_dir=args.log_dir, tb_dir=tb_dir)
    tensorboard_log = logger.tb_dir

    # Preference model pre-training (offline)
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

        # Determine observation dimensionality from a single sample
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

    # Shared lambda for SubprocVecEnv workers (important for LagPPO / RCPO)
    manager = mp.Manager()
    # This value is updated during training by the Lagrangian callback
    shared_lambda = manager.Value("d", 0.0)

    def get_lambda_shared() -> float:
        """
        Accessor for the current shared Lagrange multiplier.

        This function is passed to `make_env` so that reward shaping can use
        the *current* value of lambda inside vectorized worker processes.
        """
		
        return shared_lambda.value

    # === Environment factory ===
    def make_thunk(rank: int) -> Callable[[], Any]:
        """
        Build an initializer for a single environment instance.

        The initializer:
          - seeds the environment
          - optionally adds a geometric shield
          - optionally applies RCPO-style fixed-penalty shaping
          - optionally replaces the reward with a learned preference model

        :param rank: Index of the worker (used to offset the seed).
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
                # If using LagPPO or RCPO we supply reward shaping via lambda.
                reward_shaping_get_lambda=(
                    get_lambda_shared
                    if args.algo in ["lagppo", "rcpo"]
                    else None
                ),
            )

            if args.algo == "rcpo":
                # NOTE:
                # Here RCPO is implemented as a *simple fixed-penalty baseline*:
                # we use a RewardWrapper that shapes rewards as
                #
                #   r' = r - penalty_coef * cost
                #
                # using a constant `penalty_coef` from the CLI. A more general
                # RCPO pipeline (e.g. using `make_rcpo` from `src/algos/rcpo.py`)
                # can be integrated in future versions.

                class FixedPenalty(RewardWrapper):
                    """
                    Reward wrapper implementing r' = r - penalty_coef * cost.

                    This is a minimal RCPO-style baseline that does not
                    adapt the penalty coefficient during training. It is
                    primarily intended for comparison with LagPPO.
                    """

                    def __init__(self, env: Any, penalty: float) -> None:
                        super().__init__(env)
                        self.penalty: float = penalty

                    def step(self, action: Any):
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

    # Vectorized environment (SubprocVecEnv if more than one env).
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
        # Expose 'shared_lambda' and custom logger to the callback so it can synchronize across workers
        lag_callback.shared_lambda_value = shared_lambda
        lag_callback.custom_logger = logger

    elif args.algo == "rcpo":
        # At the moment, RCPO uses the fixed-penalty reward wrapper above and
        # a standard PPO backbone (see note in make_thunk). A more faithful
        # RCPO implementation can reuse `make_rcpo` from `src/algos/rcpo.py`
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
    )

    callbacks = [train_cb]
    if lag_callback is not None:
        callbacks.append(lag_callback)
	
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )
	
	# Save latest checkpoint for downstream evaluation
    ckpt_dir = os.path.join("checkpoints", args.env_id, args.algo, f"seed_{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest")
    model.save(latest_path)
    print(f"[train] Training done. Model saved to {latest_path}.zip")
	
	# Save config for reproducibility
    cfg = vars(args).copy()
    cfg["torch_version"] = torch.__version__
    cfg["numpy_version"] = np.__version__
    save_config(cfg, ckpt_dir)

    # Clean up vectorized environment and logger
    vec_env.close()
    logger.close()


if __name__ == "__main__":
    main()
