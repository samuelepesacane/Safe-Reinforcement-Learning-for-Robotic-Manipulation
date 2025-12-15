"""
Visualization entry point for Safe RL experiments.

This script provides a simple CLI to:
- Plot training learning curves from one or more log directories
  (each expected to contain a metrics.csv file).
- Visualize ablation summaries from the aggregated CSV produced
  by src/ablations.py.

All plotting logic lives in src.utils.plots. This module only parses arguments and delegates to those helpers.
"""

import argparse
import os
from typing import List
from .utils.plots import plot_learning_curves, plot_ablations


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for visualization.

    The user may supply:
      - one or more training log directories (--log_dirs),
      - a single ablations summary CSV (--ablations_csv),
      - and an output directory for the generated figures (--out_dir).

    Examples
    --------
    Plot learning curves from two runs:

        python -m src.visualize --log_dirs logs/run1 logs/run2 --out_dir results/plots

    Plot ablation summary:

        python -m src.visualize --ablations_csv results/ablations_summary.csv

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(
        description="Visualize Safe RL training curves and ablation results."
    )
    ap.add_argument(
        "--log_dirs",
        nargs="*",
        default=[],
        help=(
            "List of log directories containing metrics.csv "
            "files (one per training run)."
        ),
    )
    ap.add_argument(
        "--ablations_csv",
        type=str,
        default="",
        help=(
            "Path to the ablations summary CSV produced by src.ablations. "
            "If provided, ablation plots will be generated."
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/plots",
        help="Output directory where all plots will be saved.",
    )
    return ap.parse_args()


def main():
    """
    Main entry point for visualization.

    Depending on the provided arguments, this function will:

      - Call plot_learning_curves if one or more --log_dirs
        are specified. Each directory is expected to contain a
        metrics.csv file with per-run training metrics.

      - Call plot_ablations if --ablations_csv is specified.
        This CSV is typically the output of src.ablations, aggregating
        safety-performance metrics across a grid of configurations.

    All generated figures are saved under --out_dir.
    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.log_dirs:
        plot_learning_curves(
            log_dirs=args.log_dirs,
            out_dir=args.out_dir,
            title="Training",
        )

    if args.ablations_csv:
        plot_ablations(
            ablations_csv=args.ablations_csv,
            out_dir=args.out_dir,
        )

    print(f"[visualize] Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
