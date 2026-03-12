from typing import List, Optional
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Human-readable labels for known metric keys.
# Used to replace raw log keys with clean axis labels in all plots.
METRIC_LABELS = {
    "train/ep_return":              "Episode Return",
    "train/ep_cost":                "Episode Cost",
    "train/lambda":                 "Lagrange Multiplier λ",
    "train/avg_cost_per_step":          "Avg. Cost per Step",
    "train/violation_rate":             "Violation Rate",
    "train/shield_intervention_rate":   "Shield Intervention Rate",
    "eval/avg_return":              "Avg. Return",
    "eval/avg_cost":                "Avg. Episodic Cost",
    "lagrangian/lambda":            "Lagrange Multiplier λ",
    "lagrangian/avg_cost_per_step": "Avg. Cost per Step",
    "eval/violation_rate":          "Violation Rate",
    "eval/cvar_cost_alpha_0.1":     "CVaR Cost (α=0.1)",
    "eval/avg_interventions":       "Avg. Shield Interventions",
}

# Candidate metrics to look for in logged data
CANDIDATE_METRICS = [
    "train/ep_return",
    "train/ep_cost",
    "train/lambda",
    "train/avg_cost_per_step",
    "train/violation_rate",
    "train/shield_intervention_rate",
    "eval/avg_return",
    "eval/avg_cost",
    "lagrangian/lambda",
    "lagrangian/avg_cost_per_step",
    "eval/violation_rate",
    "eval/cvar_cost_alpha_0.1",
    "eval/avg_interventions",
]


def _parse_condition(basename: str):
    """
    Parse a log directory basename into (condition, algo, shield) labels.

    We expect directory names of the form <algo>_<shield>_seed<N>, e.g.
    "ppo_shield_off_seed0" or "lagppo_shield_on_seed2". The seed suffix is
    stripped so multiple seeds from the same condition share a label and
    seaborn aggregates them into a single confidence band automatically.

    :param basename: Directory name of a single training run.
        :type basename: str

    :return: Tuple of (condition, algo, shield) strings.
        :rtype: tuple[str, str, str]
    """
    # Strip trailing _seed<N> so seeds from the same condition share a label
    name = re.sub(r'_seed\d+$', '', basename)

    # Try known algo prefixes longest-first to avoid "ppo" matching before "lagppo"
    algo = "unknown"
    for candidate in ["lagppo", "rcpo", "sac", "ppo"]:
        if name.startswith(candidate):
            algo = candidate
            name = name[len(candidate):].lstrip("_")
            break

    shield = name if name else "unknown"
    condition = f"{algo}_{shield}" if algo != "unknown" else shield

    return condition, algo, shield


def _read_metrics_from_dir(ld: str) -> Optional[pd.DataFrame]:
    """
    Read metrics from a single log directory.

    Tries metrics.jsonl first (richer format written by the training logger),
    then falls back to metrics.csv. Returns None if neither file exists or
    both fail to parse.

    :param ld: Path to a log directory created by the training script.
        :type ld: str

    :return: DataFrame with all recorded metrics, or None if unavailable.
        :rtype: Optional[pd.DataFrame]
    """
    jsonl_path = os.path.join(ld, "metrics.jsonl")
    csv_path   = os.path.join(ld, "metrics.csv")
    df: Optional[pd.DataFrame] = None

    if os.path.exists(jsonl_path):
        try:
            df = pd.read_json(jsonl_path, lines=True)
        except Exception as e:
            print(f"[plots] failed to read {jsonl_path}: {e}")
            df = None

    # Fall back to CSV only if JSONL was missing or failed to parse
    if (df is None or df.empty) and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[plots] failed to read {csv_path}: {e}")
            df = None

    return df


def plot_learning_curves(
    log_dirs: List[str],
    out_dir: str,
    title: str = "Learning Curves",
) -> None:
    """
    Plot training learning curves from one or more log directories.

    Each directory basename is parsed into an (algo, shield) pair so curves
    are grouped by experimental condition. When multiple seeds share the same
    condition label, seaborn aggregates them into mean ± confidence band
    automatically.

    One PNG is saved per available metric under out_dir.

    :param log_dirs: Log directories produced by the training script.
        :type log_dirs: List[str]
    :param out_dir: Directory where PNGs are saved.
        :type out_dir: str
    :param title: Base title prepended to every plot.
        :type title: str
    """
    os.makedirs(out_dir, exist_ok=True)
    dfs = []

    for ld in log_dirs:
        df = _read_metrics_from_dir(ld)
        if df is None or df.empty:
            print(f"[plot_learning_curves] no metrics found in {ld} (skipping)")
            continue

        df = df.copy()
        basename = os.path.basename(ld)
        condition, algo, shield = _parse_condition(basename)

        df["source"] = condition 
        df["algo"]   = algo
        df["shield"] = shield
        dfs.append(df)

    if not dfs:
        print("[plot_learning_curves] no data to plot for any provided log_dirs")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    sns.set(style="whitegrid")

    for metric in CANDIDATE_METRICS:
        if metric not in df_all.columns:
            continue

        ylabel = METRIC_LABELS.get(metric, metric)

        # Shield intervention rate is only meaningful for shield_on runs
        # plotting shield_off lines alongside it would be misleading
        if metric == "eval/avg_interventions":
            plot_df = df_all[df_all["shield"] == "shield_on"]
            if plot_df.empty:
                continue
            hue_col = "algo"
            subtitle = f"{title} — {ylabel} (shield on only)"
        else:
            plot_df = df_all
            hue_col = "source"
            subtitle = f"{title} — {ylabel}"

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=plot_df,
            x="step",
            y=metric,
            hue=hue_col,
            ax=ax,
        )
        ax.set_title(subtitle)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        plt.tight_layout()

        # Replace "/" with "_" so the metric key is a safe filename
        safe_name = metric.replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot_learning_curves] saved {out_path}")


def plot_ablations(ablations_csv: str, out_dir: str) -> None:
    """
    Plot bar charts summarising ablation study results.

    One bar chart is produced per available metric, with setting on the
    x-axis and algorithm as the hue. Error bars show 95% CI across seeds.

    :param ablations_csv: Path to the aggregated ablations CSV from src.ablations.
        :type ablations_csv: str
    :param out_dir: Directory where PNGs are saved.
        :type out_dir: str
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ablations_csv):
        print(f"[plot_ablations] ablations CSV not found: {ablations_csv}")
        return

    df = pd.read_csv(ablations_csv)
    sns.set(style="whitegrid")

    bar_metrics = [
        "avg_return",
        "avg_cost",
        "violation_rate",
        "cvar_cost_alpha_0.1",
        "avg_interventions",
    ]

    for m in bar_metrics:
        if m not in df.columns:
            continue

        ylabel = METRIC_LABELS.get(m, m)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="setting",
            y=m,
            hue="algo",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(f"Ablations — {ylabel}")
        ax.set_xlabel("Setting")
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"ablations_{m}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot_ablations] saved {out_path}")
