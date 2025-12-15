"""
Evaluate a trained policy with safety-aware metrics.

Loads a PPO or SAC checkpoint, runs a fixed number of episodes on the raw environment (no shaping/shield), 
and saves summary stats to CSV (metrics.csv) under the directory specified by '--log_dir'.
"""

import os
import argparse
from typing import Dict, Any, List
from stable_baselines3 import PPO, SAC
from .envs.make_env import make_env
from .safety.metrics import aggregate_episode_metrics, dump_metrics_csv


def parse_args() -> argparse.Namespace:
    """
    CLI flags for evaluation (env, model path, episodes, logging).

    :return: Parsed command-line arguments.
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
            "Path to the trained SB3 model. You may omit the '.zip' extension; "
            "it will be added automatically if missing."
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
        help="Directory where 'metrics.csv' will be saved.",
    )
    return ap.parse_args()


def detect_algo_from_path(path: str) -> str:
    """
    Infer the algorithm family (PPO or SAC) from the checkpoint path.

    The detection is based on substring matching:
    - If 'sac' appears in the lowercased path, we assume a SAC model
    - If 'ppo' appears, we assume a PPO model
    - Otherwise we default to PPO

    This works because:
    - LagPPO and RCPO both use a PPO backbone and are saved via PPO.save(...),
      so their checkpoints can be loaded with PPO
    - Plain SAC runs are saved under directories containing 'sac'

    :param path: Filesystem path (directory or file) pointing to the checkpoint
    :type path: str

    :return: Detected algorithm identifier, either "ppo" or "sac"
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
    Main function for evaluation.

    Steps:
      1. Parse CLI arguments and create the evaluation environment
      2. Load the trained PPO/SAC model from disk
      3. Roll out the deterministic policy for the requested number of episodes
      4. For each episode, record:
         - cumulative reward (returns)
         - cumulative safety cost (cost)
         - episode length (length)
         - number of shield interventions (interventions)
         - a binary success flag (success)
      5. Aggregate all episodes using function aggregate_episode_metrics and
         save the resulting metrics to metrics.csv

    The environment is created via make_env so that step outputs and
    safety-cost handling are consistent with the training pipeline
    """
	
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # Create a single evaluation environment. We do not attach shields or
    # Lagrangian reward shaping here; the goal is to measure true environment
    # reward and cost, not shaped objectives
    env = make_env(args.env_id, seed=args.seed)

    # Detect which SB3 class to use for loading the model.
    algo = detect_algo_from_path(args.model_path)
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
        # Gymnasium reset returns (obs, info) and we use a different seed per episode for reproducibility while still allowing variability
        obs, info = env.reset(seed=args.seed + ep)

        done = False
        ep_ret = 0.0
        ep_cost = 0.0
        length = 0
        interventions = 0
        success = False

        while not done:
            # Deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)

            # The environment is wrapped by make_env, so we expect the standard
            # Gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
            obs, r, terminated, truncated, info = env.step(action)

            ep_ret += float(r)
            ep_cost += float(info.get("cost", 0.0))
            length += 1

            # If a shield such as GenericKeepoutShield was used in training or
            # deployment, it can mark interventions via this flag
            if info.get("shield_intervened", False):
                interventions += 1

            # Try to detect success when the environment exposes a flag
            if (
                info.get("is_success", False)
                or info.get("success", False)
                or info.get("goal_achieved", False)
            ):
                success = True

            if args.render:
                # Rendering is optional and may not be implemented in all envs
                try:
                    env.render()
                except Exception:
                    pass

            done = bool(terminated or truncated)

        # NOTE: aggregate_episode_metrics expects the keys:
        # "returns", "cost", "length", "interventions", "success"
        episodes.append(
            dict(
                returns=ep_ret,
                cost=ep_cost,
                length=length,
                interventions=interventions,
                success=success,
            )
        )

    # Aggregate metrics across episodes and save them as a single-row CSV
    metrics = aggregate_episode_metrics(episodes)
    out_csv = os.path.join(args.log_dir, "metrics.csv")
    dump_metrics_csv(metrics, out_csv)

    print("[evaluate] Aggregated evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"[evaluate] Saved metrics to {out_csv}")

    # Close environment to free resources
    env.close()


if __name__ == "__main__":
    main()
