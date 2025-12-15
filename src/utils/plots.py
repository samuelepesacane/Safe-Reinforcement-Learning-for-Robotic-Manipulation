from typing import List, Optional
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _read_metrics_from_dir(ld: str) -> Optional[pd.DataFrame]:
    """
    Read metrics from a single logging directory.

    The function first attempts to load ``metrics.jsonl`` (JSON-lines format),
    and falls back to ``metrics.csv`` if JSON is unavailable or unreadable.
    If no usable file is found, it returns ``None``.

    :param ld: Path to a log directory created by the training script.
        :type ld: str

    :return: DataFrame containing all recorded metrics for that directory,
        or ``None`` if no data could be loaded.
        :rtype: Optional[pandas.DataFrame]
    """
    jsonl_path = os.path.join(ld, "metrics.jsonl")
    csv_path = os.path.join(ld, "metrics.csv")
    df: Optional[pd.DataFrame] = None

    if os.path.exists(jsonl_path):
        try:
            df = pd.read_json(jsonl_path, lines=True)
        except Exception as e:
            print(f"[plots] failed to read {jsonl_path}: {e}")
            df = None

    if (df is None or df.empty) and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[plots] failed to read {csv_path}: {e}")
            df = None

    return df


def plot_learning_curves(log_dirs: List[str], out_dir: str, title: str = "Learning Curves") -> None:
    """
    Plot learning curves for a set of candidate metrics from one or more log directories.

    For each directory, the function reads ``metrics.jsonl`` or ``metrics.csv``,
    adds a ``source`` column indicating the run, concatenates everything, and
    then produces line plots for several common metrics (returns, costs,
    lambda, violation rates, etc.). Each metric is saved as a separate PNG.

    :param log_dirs: List of log directories containing metrics files produced
        by the training script.
        :type log_dirs: List[str]
    :param out_dir: Directory where the generated PNG plots will be saved.
        :type out_dir: str
    :param title: Base title for the plots (metric name is appended).
        :type title: str

    :return: This function does not return anything; it saves figures to disk.
        :rtype: None
    """
    os.makedirs(out_dir, exist_ok=True)
    dfs = []

    for ld in log_dirs:
        df = _read_metrics_from_dir(ld)
        if df is None or df.empty:
            print(f"[plot_learning_curves] no metrics found in {ld} (skipping)")
            continue
        df = df.copy()
        df["source"] = os.path.basename(ld)
        dfs.append(df)

    if not dfs:
        print("[plot_learning_curves] no data to plot for any provided log_dirs")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Flexible candidate metrics to look for (covers train/eval/lagrangian keys)
    candidate_metrics = [
        "train/ep_return",
        "train/ep_cost",
        "train/lambda",
        "eval/avg_return",
        "eval/avg_cost",
        "lagrangian/lambda",
        "lagrangian/avg_cost_per_step",
        "eval/violation_rate",
        "eval/cvar_cost_alpha_0.1",
    ]

    for metric in candidate_metrics:
        if metric not in df_all.columns:
            continue
        plt.clf()
        sns.lineplot(data=df_all, x="step", y=metric, hue="source")
        plt.title(f"{title} - {metric}")
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.tight_layout()
        safe_name = metric.replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_name}.png")
        plt.savefig(out_path)
        print(f"[plot_learning_curves] saved {out_path}")

    plt.clf()


def plot_ablations(ablations_csv: str, out_dir: str) -> None:
    """
    Plot bar charts summarizing ablation study results.

    The input CSV is expected to contain one row per experimental setting
    (e.g. shield on/off, different cost budgets) and columns for aggregate
    metrics such as average return, average cost, violation rate, and
    CVaR-type statistics. For each available metric, a bar plot is created
    comparing settings and algorithms.

    :param ablations_csv: Path to the aggregated ablations CSV produced by
        ``ablations.py``.
        :type ablations_csv: str
    :param out_dir: Directory where the ablation plots will be saved.
        :type out_dir: str

    :return: This function does not return anything; it saves figures to disk.
        :rtype: None
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(ablations_csv):
        print(f"[plot_ablations] ablations CSV not found: {ablations_csv}")
        return

    df = pd.read_csv(ablations_csv)
    sns.set(style="whitegrid")

    metrics = [
        "avg_return",
        "avg_cost",
        "violation_rate",
        "cvar_cost_alpha_0.1",
        "avg_interventions",
    ]

    for m in metrics:
        if m not in df.columns:
            continue

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="setting",
            y=m,
            hue="algo",
            errorbar=("ci", 95),
        )
        plt.title(f"Ablations - {m}")
        plt.xlabel("setting")
        plt.ylabel(m)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"ablations_{m}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path)
        print(f"[plot_ablations] saved {out_path}")
        plt.close()
