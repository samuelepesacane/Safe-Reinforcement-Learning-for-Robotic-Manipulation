"""
Visualization entry point for Safe RL experiments.

Provides a simple CLI to plot training learning curves from one or more log
directories, or to visualize ablation summaries from the aggregated CSV
produced by src/ablations.py.

All plotting logic lives in src.utils.plots. This script only parses
arguments and delegates to those helpers.
"""

import argparse
import os
from typing import List
from .utils.plots import plot_learning_curves, plot_ablations


def parse_args() -> argparse.Namespace:
    """
    Parse CLI flags for visualization.

    Supply one or more training log directories, an ablations summary CSV,
    or both. All generated figures are saved under --out_dir.

    :return: Parsed args.
        :rtype: argparse.Namespace
    """
    ap = argparse.ArgumentParser(
        description="Visualize Safe RL training curves and ablation results."
    )
    ap.add_argument(
        "--log_dirs",
        nargs="*",
        default=[],
        help="List of log directories, each containing a metrics.csv file.",
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
    Generate plots from training logs or ablation results.

    Calls plot_learning_curves if --log_dirs are specified, and
    plot_ablations if --ablations_csv is specified. Both can be used
    together in a single call.
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
