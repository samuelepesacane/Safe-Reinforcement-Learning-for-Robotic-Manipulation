"""
Evaluate a trained policy with safety-aware metrics.

Loads a PPO or SAC checkpoint, runs a fixed number of episodes on the raw
environment (no reward shaping or shield), and saves summary stats to
metrics.csv under the directory specified by --log_dir.

No shield is applied during evaluation by design: the goal is to measure
the true safety behavior of the learned policy, not the behavior of the
policy plus a runtime correction layer.
"""

import os
import argparse
from typing import Dict, Any, List
from stable_baselines3 import PPO, SAC
from .envs.make_env import make_env
from .safety.metrics import aggregate_episode_metrics, dump_metrics_csv


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

    # No shield or reward shaping: we want to measure what the policy learned,
    # not what the policy plus a correction layer does
    env = make_env(args.env_id, seed=args.seed)

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
        success = False

        while not done:
            # Deterministic policy: no exploration noise during evaluation
            action, _ = model.predict(obs, deterministic=True)

            obs, r, terminated, truncated, info = env.step(action)

            ep_ret += float(r)
            ep_cost += float(info.get("cost", 0.0))
            length += 1

            # Shield interventions are zero here by design (no shield at eval),
            # but we track the flag in case the env wrapper sets it for logging
            if info.get("shield_intervened", False):
                interventions += 1

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
                success=success,
            )
        )

    metrics = aggregate_episode_metrics(episodes)
    out_csv = os.path.join(args.log_dir, "metrics.csv")
    dump_metrics_csv(metrics, out_csv)

    print("[evaluate] Aggregated evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"[evaluate] Saved metrics to {out_csv}")

    env.close()


if __name__ == "__main__":
    main()
