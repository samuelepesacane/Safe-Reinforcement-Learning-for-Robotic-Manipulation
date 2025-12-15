"""
Small driver script for ablation runs.

Builds a grid over a few safety knobs (shield, budgets, lambda LR, preferences, algo), 
calls src.train / src.evaluate in subprocesses, and aggregates all metrics into one CSV.

Launch and aggregate ablation studies for Safe RL experiments.
So, this file is the outer loop that systematically probes the safety-performance trade-offs of the Safe RL stack described in the project.
"""

import argparse
import os
import subprocess
# import itertools
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args() -> argparse.Namespace:
    """
    CLI flags for ablations (seeds, timesteps, output CSV, parallelism).

    :return: Parsed command-line arguments.
        :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(
        description="Run ablations and aggregate Safe RL results"
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="List of random seeds to run for each setting.",
    )
    ap.add_argument(
        "--total_timesteps",
        type=int,
        default=150000,
        help="Number of training timesteps per run.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="results/ablations_summary.csv",
        help="Path to the final aggregated CSV file.",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of training jobs to run in parallel.",
    )
    return ap.parse_args()


def run_cmd(cmd: str, log_file: Optional[str] = None) -> None:
    """
    Run a shell command, optionally logging stdout/stderr to a file.

    Training and evaluation are launched as subprocesses to keep
    this script simple and decoupled from the Python APIs of src.train and
    src.evaluate. If any command fails, a CalledProcessError is raised.

    :param cmd: Shell command to execute.
        :type cmd: str
    :param log_file: Optional path to a file where stdout/stderr should be
        redirected. If None, the output is inherited from the parent
        process (i.e. printed to the console).
        :type log_file: Optional[str]

    :return: This function does not return anything; it raises on error.
        :rtype: None
    """
    print(f"[Run] {cmd}")
    if log_file is not None:
        with open(log_file, "w") as f:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
    else:
        subprocess.run(cmd, shell=True, check=True)


def main():
    """
    Main entry point for the ablation launcher.

    High-level process:
      1. Define an experiment grid over:
         - shield on/off (LagPPO),
         - Lagrangian learning rate (lr_lambda),
         - cost budget (cost_budget),
         - preference-based rewards on/off, and
         - algorithm choice (PPO, SAC, RCPO baseline, LagPPO).
      2. For each configuration and random seed, build:
         - a training command (python -m src.train),
         - an evaluation command (python -m src.evaluate),
         and store them in a job list.
      3. Execute the training (optionally in parallel) followed by evaluation.
      4. Collect the resulting metrics.csv files into a single DataFrame
         and write args.out_csv.

    This script assumes:
      - checkpoints are saved by src.train under
        checkpoints/<env_id>/<algo>/seed_<seed>/latest, and
      - src.evaluate writes a`metrics.csv in each evaluation directory.
    """
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    # For now we fix the ablation environment. This matches the configuration described in the README
    env_id: str = "SafetyPointPush1-v0"  # If needed this can be turned into a CLI flag
    base_log: str = "logs/ablations"

    settings: List[Dict[str, Any]] = []

    # Experiment grid over safety knobs and algorithm choice

    # Shield on/off (with LagPPO)
    for shield in [False, True]:
        settings.append(
            dict(
                name=f"shield_{'on' if shield else 'off'}",
                algo="lagppo",
                use_shield=shield,
            )
        )

    # Lagrangian learning rate (dual ascent step size)
    for lr in [1e-4, 5e-4, 1e-3]:
        settings.append(
            dict(
                name=f"lr_lambda_{lr}",
                algo="lagppo",
                lr_lambda=lr,
            )
        )

    # Cost budget (average per-step constraint)
    for budget in [0.01, 0.05, 0.1]:
        settings.append(
            dict(
                name=f"budget_{budget}",
                algo="lagppo",
                cost_budget=budget,
            )
        )

    # Preference-based reward vs environment reward (LagPPO)
    for use_pref in [False, True]:
        settings.append(
            dict(
                name=f"prefs_{'on' if use_pref else 'off'}",
                algo="lagppo",
                use_preferences=use_pref,
            )
        )

    # Algorithm comparison (PPO, SAC, RCPO baseline, LagPPO)
    for algo in ["ppo", "sac", "rcpo", "lagppo"]:
        settings.append(
            dict(
                name=f"algo_{algo}",
                algo=algo,
            )
        )

    # Prepare all jobs (train + eval)

    # Each job is a tuple: (setting_dict, seed, train_cmd, eval_cmd, eval_dir)
    jobs: List[Tuple[Dict[str, Any], int, str, str, str]] = []

    for s in settings:
        for seed in args.seeds:
            log_dir = os.path.join(base_log, s["name"], f"seed_{seed}")
            ckpt = os.path.join(
                "checkpoints",
                env_id,
                s.get("algo", "lagppo"),
                f"seed_{seed}",
                "latest",
            )
            eval_dir = os.path.join(
                "results",
                "ablations_eval",
                s["name"],
                f"seed_{seed}",
            )
            os.makedirs(eval_dir, exist_ok=True)

            # Build training command consistent with src.train.parse_args
            train_cmd = (
                f"python -m src.train --env_id {env_id} "
                f"--algo {s.get('algo', 'lagppo')} "
                f"--total_timesteps {args.total_timesteps} "
                f"--seed {seed} --log_dir {log_dir}"
            )
            if s.get("use_shield", False):
                train_cmd += " --use_shield"
            if "lr_lambda" in s:
                train_cmd += f" --lr_lambda {s['lr_lambda']}"
            if "cost_budget" in s:
                train_cmd += f" --cost_budget {s['cost_budget']}"
            if s.get("use_preferences", False):
                # Fixed number of preference steps for this study
                train_cmd += " --use_preferences --pref_steps 20000"

            # Build evaluation command consistent with src.evaluate.parse_args
            eval_cmd = (
                f"python -m src.evaluate --env_id {env_id} "
                f"--model_path {ckpt} --episodes 10 --seed {seed} "
                f"--log_dir {eval_dir}"
            )

            jobs.append((s, seed, train_cmd, eval_cmd, eval_dir))

    # Execute jobs (with optional parallelism)

    if args.parallel > 1:
        # In parallel mode, we parallelize training runs and then evaluate each job as soon as its training finishes.
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_cmd, j[2]): j for j in jobs}
            for f in as_completed(futures):
                s, seed, _, eval_cmd, _ = futures[f]
                # Propagate any training error
                f.result()
                # Evaluate sequentially after training completes
                run_cmd(eval_cmd)
    else:
        # Sequential mode: train then evaluate for each job
        for s, seed, train_cmd, eval_cmd, _ in jobs:
            run_cmd(train_cmd)
            run_cmd(eval_cmd)

    # Aggregate results from all evaluation directories

    records: List[Dict[str, Any]] = []
    for s, seed, _, _, eval_dir in jobs:
        metrics_csv = os.path.join(eval_dir, "metrics.csv")
        if os.path.exists(metrics_csv):
            try:
                df = pd.read_csv(metrics_csv)
            except Exception as exc:
                print(f"[Ablations] Failed to read {metrics_csv}: {exc}")
                continue

            if not df.empty:
                rec = df.iloc[0].to_dict()
                rec.update(
                    dict(
                        setting=s["name"],
                        algo=s.get("algo", "lagppo"),
                        seed=seed,
                    )
                )
                records.append(rec)

    if records:
        df_all = pd.DataFrame.from_records(records)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df_all.to_csv(args.out_csv, index=False)
        print(f"[Ablations] Saved aggregated summary to {args.out_csv}")
    else:
        print("[Ablations] No records collected; check that evaluations succeeded.")


if __name__ == "__main__":
    main()
