"""
Evaluate a trained policy with safety-aware metrics.

Loads a PPO or SAC checkpoint, runs a fixed number of episodes on the
environment (no reward shaping; shield OFF unless --use_shield is passed),
and saves summary stats to metrics.csv under the directory specified by
--log_dir.

Shield defaults to OFF, preserving every existing result in this project:
until 2026-09-11 this module never applied the shield at all, regardless of
whether the checkpoint being evaluated was trained with one (D15 in
CODE_AUDIT_AND_EXECUTION_PLAN.md) -- every historical "shield_on" eval row,
including the extended README's three-way table, measures the converged
policy running WITHOUT the shield, not shielded deployment. That is a
legitimate question (what did the policy learn) but a different one from
"what happens when this policy is actually deployed behind the shield",
which --use_shield now answers. Passing it does not change any existing
number; it only makes previously-unmeasured shielded-deployment runs
possible, into their own --log_dir.
"""

import os
import argparse
from typing import Dict, Any, List
from stable_baselines3 import PPO, SAC
from .envs.make_env import make_env
from .safety.metrics import aggregate_episode_metrics, dump_metrics_csv
from .train import build_shield_factory


def parse_args() -> argparse.Namespace:
    """
    Parse CLI flags for evaluation.

    We expose env, model path, episode count, seed, and output directory
    so evaluation runs are fully reproducible from the command line.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(
        description="Evaluate a trained policy with safety-aware metrics."
    )
    ap.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Gymnasium/Safety-Gymnasium environment ID.",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the trained SB3 model. The '.zip' extension is added "
            "automatically if missing."
        ),
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes to run.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for environment resets.",
    )
    ap.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation (if supported).",
    )
    ap.add_argument(
        "--log_dir",
        type=str,
        default="results/eval",
        help="Directory where metrics.csv will be saved.",
    )
    ap.add_argument(
        "--use_shield",
        action="store_true",
        help=(
            "Apply the shield during evaluation (default OFF, preserving every "
            "existing result -- see the module docstring / D15). Only meaningful "
            "when the checkpoint was trained with a compatible action space; the "
            "shield type/params below should match what the checkpoint was "
            "trained with if you want a like-for-like deployment measurement."
        ),
    )
    ap.add_argument(
        "--shield_type",
        type=str,
        default="geometric",
        choices=["geometric", "riemannian"],
        help="Which shield to use when --use_shield is set. Mirrors train.py.",
    )
    ap.add_argument(
        "--shield_alpha",
        type=float,
        default=0.1,
        help="Gradient scaling coefficient for the Riemannian shield. Mirrors train.py.",
    )
    ap.add_argument(
        "--shield_influence_radius",
        type=float,
        default=0.5,
        help="Influence radius for the Riemannian shield. Mirrors train.py.",
    )
    ap.add_argument(
        "--shield_kinematic_model",
        type=str,
        default="world_xy",
        choices=["world_xy", "heading_fit"],
        help="Shield next-position prediction model. Mirrors train.py.",
    )
    return ap.parse_args()


def detect_algo_from_path(path: str) -> str:
    """
    Infer the algorithm family (PPO or SAC) from the checkpoint path.

    Detection is based on substring matching in the lowercased path.
    LagPPO and RCPO both use a PPO backbone and are saved with PPO.save(),
    so their checkpoints load correctly with PPO. Plain SAC runs are saved
    under directories containing 'sac'. We default to PPO when neither
    substring is found.

    :param path: Filesystem path pointing to the checkpoint.
        :type path: str

    :return: Either "ppo" or "sac".
        :rtype: str
    """
    p = path.lower()
    if "sac" in p:
        return "sac"
    if "ppo" in p:
        return "ppo"
    return "ppo"


def main():
    """
    Evaluate a trained checkpoint and save safety-aware metrics to CSV.

    Steps:
      1. Parse CLI arguments and create a single evaluation environment
      2. Load the trained PPO or SAC model from disk
      3. Roll out the deterministic policy for the requested number of episodes
      4. Record cumulative reward, cumulative cost, episode length,
         shield interventions, and success flag for each episode
      5. Aggregate across episodes and save to metrics.csv
    """
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # No reward shaping either way. Shield defaults OFF (see module docstring
    # / D15); --use_shield wires the same factory train.py uses, so a shielded
    # eval reflects the actual deployment condition rather than a bespoke one.
    shield_factory = (
        build_shield_factory(
            args.env_id,
            shield_type=args.shield_type,
            alpha=args.shield_alpha,
            influence_radius=args.shield_influence_radius,
            kinematic_model=args.shield_kinematic_model,
        )
        if args.use_shield
        else None
    )
    env = make_env(
        args.env_id,
        seed=args.seed,
        use_shield=args.use_shield,
        shield_factory=shield_factory,
    )

    algo = detect_algo_from_path(args.model_path)
    # SB3 accepts paths with or without the .zip extension, but we normalise
    # to always include it so the path is unambiguous on disk
    model_path = (
        args.model_path
        if args.model_path.endswith(".zip")
        else args.model_path + ".zip"
    )

    if algo == "sac":
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)

    episodes: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        # Different seed per episode for variability while remaining reproducible
        obs, info = env.reset(seed=args.seed + ep)

        done = False
        ep_ret = 0.0
        ep_cost = 0.0
        length = 0
        interventions = 0
        gradient_interventions = 0
        success = False

        while not done:
            # Deterministic policy: no exploration noise during evaluation
            action, _ = model.predict(obs, deterministic=True)

            obs, r, terminated, truncated, info = env.step(action)

            ep_ret += float(r)
            ep_cost += float(info.get("cost", 0.0))
            length += 1

            # With --use_shield these are no longer zero by construction (D15
            # fix); without it they stay zero exactly as before, since
            # ShieldingActionWrapper is never built and never sets these keys.
            if info.get("shield_intervened", False):
                interventions += 1
                # A gradient-stage intervention (RiemannianShield only); its
                # absence for an intervening step means the step was
                # bisection-only -- see the trap analysis in SESSION_HANDOFF.md.
                if info.get("shield_gradient_intervened", False):
                    gradient_interventions += 1

            # Safety-Gymnasium and Gymnasium-Robotics expose success under
            # different keys; we check all three to be safe
            if (
                info.get("is_success", False)
                or info.get("success", False)
                or info.get("goal_achieved", False)
            ):
                success = True

            if args.render:
                try:
                    env.render()
                except Exception:
                    pass

            done = bool(terminated or truncated)

        # aggregate_episode_metrics expects these exact keys
        episodes.append(
            dict(
                returns=ep_ret,
                cost=ep_cost,
                length=length,
                interventions=interventions,
                gradient_interventions=gradient_interventions,
                success=success,
            )
        )

    metrics = aggregate_episode_metrics(episodes)

    # Derived rates, useful specifically for a shielded eval (0 and undefined
    # -> 0 when shield is off, matching every historical run's silent zeros).
    avg_len = metrics.get("avg_len", 0.0)
    avg_interventions = metrics.get("avg_interventions", 0.0)
    avg_gradient_interventions = metrics.get("avg_gradient_interventions", 0.0)
    metrics["intervention_rate"] = (
        avg_interventions / avg_len if avg_len > 0 else 0.0
    )
    metrics["bisection_only_fraction"] = (
        (avg_interventions - avg_gradient_interventions) / avg_interventions
        if avg_interventions > 0 else 0.0
    )

    out_csv = os.path.join(args.log_dir, "metrics.csv")
    dump_metrics_csv(metrics, out_csv)

    print("[evaluate] Aggregated evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"[evaluate] Saved metrics to {out_csv}")

    env.close()


if __name__ == "__main__":
    main()
