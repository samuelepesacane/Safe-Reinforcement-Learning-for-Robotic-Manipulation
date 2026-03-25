"""
Plot lambda trajectories, cost curves, and shield intervention rates
for the three-way shield comparison (no shield, geometric, Riemannian)
across three environments (Push, Goal, Car) for LagPPO.

Produces three figures:
  1. fig_lambda.pdf  -- lambda over training, 3 environments x 3 conditions
  2. fig_cost.pdf    -- per-step cost over training, same layout
  3. fig_intervention.pdf -- shield intervention rate over training,
                             geometric vs Riemannian only

Each curve is the mean across 3 seeds with a shaded 95% confidence interval.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple

# Configuration

RESULTS_DIR = "results"
LOGS_DIR    = "logs"
OUT_DIR     = "results/plots_three_way"
os.makedirs(OUT_DIR, exist_ok=True)

# Smoothing window applied to noisy per-rollout lambda series
SMOOTH_WINDOW = 20

# Colors and labels for the three shield conditions
SHIELD_STYLES = {
    "none":       {"color": "#4C72B0", "label": "No shield",        "ls": "--"},
    "geometric":  {"color": "#DD8452", "label": "Geometric shield", "ls": "-."},
    "riemannian": {"color": "#55A868", "label": "Riemannian shield","ls": "-"},
}

ENV_LABELS = {
    "push": "SafetyPointPush1-v0\n(2 hazards, point robot)",
    "goal": "SafetyPointGoal1-v0\n(8 hazards, point robot)",
    "car":  "SafetyCarGoal1-v0\n(8 hazards, car robot)",
}

# Log directory name patterns for each (env, shield, seed) combination
# Format: logs/<prefix>_seed<n>/metrics.jsonl
LOG_PATTERNS = {
    ("push", "none"):       "logs/ppo_shield_off_seed{seed}",       # placeholder; push none uses original logs
    ("push", "none"):       "logs/lagppo_shield_off_seed{seed}",
    ("push", "geometric"):  "logs/lagppo_shield_on_seed{seed}",
    ("push", "riemannian"): "logs/push_riemannian_lagppo_seed{seed}",
    ("goal", "none"):       "logs/goal_lagppo_shield_off_seed{seed}",
    ("goal", "geometric"):  "logs/goal_lagppo_shield_on_seed{seed}",
    ("goal", "riemannian"): "logs/goal_riemannian_lagppo_seed{seed}",
    ("car",  "none"):       "logs/car_lagppo_shield_off_seed{seed}",
    ("car",  "geometric"):  "logs/car_lagppo_shield_on_seed{seed}",
    ("car",  "riemannian"): "logs/car_riemannian_lagppo_seed{seed}",
}

SEEDS = [0, 1, 2]

# Data loading helpers

def load_jsonl(path: str) -> List[Dict]:
    """
    Load a metrics.jsonl file into a list of dicts.

    We read every line independently so a partial write at the end of a
    run does not crash the loader.

    :param path: Path to the .jsonl file.
        :type path: str

    :return: List of parsed JSON objects, one per line.
        :rtype: List[Dict]
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def extract_series(
    rows: List[Dict],
    key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a (steps, values) pair for a given metric key from a log.

    Rows that do not contain the key are skipped, so lagrangian/* and
    train/* series can be extracted from the same file without issue.

    :param rows: Parsed log rows from load_jsonl.
        :type rows: List[Dict]
    :param key: Metric key to extract (e.g. "lagrangian/lambda").
        :type key: str

    :return: Arrays of (steps, values), both float32.
        :rtype: Tuple[np.ndarray, np.ndarray]
    """
    steps, vals = [], []
    for row in rows:
        if key in row and "step" in row:
            steps.append(row["step"])
            vals.append(float(row[key]))
    return np.array(steps, dtype=np.float32), np.array(vals, dtype=np.float32)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a uniform moving average to a 1D array.

    Uses 'valid' mode so the output is shorter than the input by
    (window - 1) elements. The caller must account for this when
    aligning with the steps array.

    :param values: Input signal.
        :type values: np.ndarray
    :param window: Moving average window size.
        :type window: int

    :return: Smoothed signal.
        :rtype: np.ndarray
    """
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_seed_series(
    env: str,
    shield: str,
    key: str,
    smooth_window: int = 1,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and aggregate a metric series across all seeds for one condition.

    Interpolates each seed's series onto a common step grid (the union of
    all step values) before computing mean and std. Returns None if no
    log files are found for this condition.

    :param env: Environment key ("push", "goal", "car").
        :type env: str
    :param shield: Shield condition ("none", "geometric", "riemannian").
        :type shield: str
    :param key: Metric key to extract.
        :type key: str
    :param smooth_window: Moving average window applied before aggregation.
        :type smooth_window: int

    :return: Tuple of (steps, mean, std) arrays, or None if no data found.
        :rtype: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    pattern = LOG_PATTERNS.get((env, shield))
    if pattern is None:
        return None

    all_steps_list = []
    all_vals_list  = []

    for seed in SEEDS:
        path = os.path.join(pattern.format(seed=seed), "metrics.jsonl")
        if not os.path.exists(path):
            continue
        rows = load_jsonl(path)
        steps, vals = extract_series(rows, key)
        if len(steps) == 0:
            continue
        if smooth_window > 1:
            vals  = smooth(vals, smooth_window)
            # Align steps to the smoothed output length
            trim  = (len(steps) - len(vals)) // 2
            steps = steps[trim: trim + len(vals)]
        all_steps_list.append(steps)
        all_vals_list.append(vals)

    if not all_vals_list:
        return None

    # Interpolate onto the shortest common step range
    max_start = max(s[0]  for s in all_steps_list)
    min_end   = min(s[-1] for s in all_steps_list)
    grid      = np.linspace(max_start, min_end, 200)

    interp_vals = []
    for steps, vals in zip(all_steps_list, all_vals_list):
        interp_vals.append(np.interp(grid, steps, vals))

    arr  = np.array(interp_vals)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    return grid, mean, std


# Plotting helpers

def plot_metric(
    axs,
    metric_key: str,
    ylabel: str,
    smooth_window: int = 1,
    scale: float = 1.0,
    ylim: Optional[Tuple] = None,
) -> None:
    """
    Fill a row of axes with one metric across all environments and shield conditions.

    :param axs: Array of matplotlib Axes, one per environment.
    :param metric_key: The metrics.jsonl key to plot.
    :param ylabel: Y-axis label for the leftmost panel.
    :param smooth_window: Smoothing window for the series.
    :param scale: Multiply values by this factor (e.g. 1e3 to convert to milli-units).
    :param ylim: Optional (ymin, ymax) tuple.
    """
    envs = ["push", "goal", "car"]
    shields = ["none", "geometric", "riemannian"]

    for col, env in enumerate(envs):
        ax = axs[col]
        any_data = False
        for shield in shields:
            result = load_seed_series(env, shield, metric_key, smooth_window)
            if result is None:
                continue
            steps, mean, std = result
            style = SHIELD_STYLES[shield]
            mean = mean * scale
            std  = std  * scale
            ax.plot(
                steps / 1e6, mean,
                color=style["color"],
                ls=style["ls"],
                lw=1.8,
                label=style["label"],
            )
            ax.fill_between(
                steps / 1e6,
                mean - std,
                mean + std,
                color=style["color"],
                alpha=0.15,
            )
            any_data = True

        ax.set_xlabel("Training steps (M)", fontsize=10)
        ax.set_title(ENV_LABELS[env], fontsize=9)
        ax.axhline(0.05 if "cost" in metric_key else 0, color="grey",
                   lw=0.8, ls=":", alpha=0.6)
        if ylim:
            ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)


# Figure 1: Lambda trajectory

def plot_lambda() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(
        "Lagrange multiplier (λ) over training — LagPPO",
        fontsize=12, fontweight="bold"
    )
    plot_metric(
        axs,
        metric_key="lagrangian/lambda",
        ylabel="λ (Lagrange multiplier)",
        smooth_window=SMOOTH_WINDOW,
        ylim=None,
    )
    handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for v in SHIELD_STYLES.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT_DIR, "fig_lambda.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# Figure 2: Per-step cost over training

def plot_cost() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(
        "Per-step cost over training — LagPPO",
        fontsize=12, fontweight="bold"
    )
    plot_metric(
        axs,
        metric_key="train/avg_cost_per_step",
        ylabel="Avg cost per step",
        smooth_window=SMOOTH_WINDOW // 2,
        ylim=None,
    )
    # Add budget line annotation on first panel
    axs[0].axhline(0.05, color="red", lw=1.0, ls=":", alpha=0.8,
                   label="Cost budget (0.05)")
    handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for v in SHIELD_STYLES.values()
    ]
    handles.append(plt.Line2D([0], [0], color="red", lw=1.0, ls=":",
                               label="Cost budget"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT_DIR, "fig_cost.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# Figure 3: Shield intervention rate

def plot_intervention() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(
        "Shield intervention rate over training — LagPPO",
        fontsize=12, fontweight="bold"
    )
    envs = ["push", "goal", "car"]
    shields_with_shield = ["geometric", "riemannian"]

    for col, env in enumerate(envs):
        ax = axs[col]
        for shield in shields_with_shield:
            result = load_seed_series(
                env, shield,
                "train/shield_intervention_rate",
                smooth_window=SMOOTH_WINDOW // 2,
            )
            if result is None:
                continue
            steps, mean, std = result
            style = SHIELD_STYLES[shield]
            ax.plot(
                steps / 1e6, mean * 100,
                color=style["color"], ls=style["ls"],
                lw=1.8, label=style["label"],
            )
            ax.fill_between(
                steps / 1e6,
                (mean - std) * 100,
                (mean + std) * 100,
                color=style["color"], alpha=0.15,
            )
        ax.set_xlabel("Training steps (M)", fontsize=10)
        ax.set_title(ENV_LABELS[env], fontsize=9)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Intervention rate (%)", fontsize=10)

    handles = [
        mpatches.Patch(color=SHIELD_STYLES[s]["color"],
                       label=SHIELD_STYLES[s]["label"])
        for s in shields_with_shield
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT_DIR, "fig_intervention.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# Figure 4: Summary bar chart (eval metrics)

def plot_summary_bars() -> None:
    """
    Bar chart of seed-averaged per-step cost and CVaR at eval time,
    LagPPO only, three environments, three shield conditions.
    Uses the aggregated numbers from evaluate.py runs.
    """
    # Seed-averaged eval metrics (from the analysis already computed)
    eval_data = {
        #  (env, shield): (avg_return, per_step_cost, cvar)
        ("push", "none"):       (0.783, 0.0498, 314.8),
        ("push", "geometric"):  (0.836, 0.0348, 285.8),
        ("push", "riemannian"): (0.666, 0.0240, 184.4),
        ("goal", "none"):       (25.980, 0.0458,  95.2),
        ("goal", "geometric"):  (26.407, 0.0517, 107.5),
        ("goal", "riemannian"): (26.425, 0.0479, 115.3),
        ("car",  "none"):       (28.790, 0.0642, 178.3),
        ("car",  "geometric"):  (28.915, 0.0609, 151.2),
        ("car",  "riemannian"): (29.596, 0.0509, 121.3),
    }

    envs    = ["push", "goal", "car"]
    shields = ["none", "geometric", "riemannian"]
    env_labels = ["Push\n(2 haz, point)", "Goal\n(8 haz, point)", "Car\n(8 haz, car)"]

    x      = np.arange(len(envs))
    width  = 0.22
    offset = [-1, 0, 1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Eval metrics — LagPPO, seed-averaged (3 seeds)",
        fontsize=12, fontweight="bold"
    )

    for ax, metric_idx, ylabel, title in [
        (axs[0], 1, "Per-step cost", "Per-step cost (lower is better)"),
        (axs[1], 2, "CVaR (α=0.1)",  "CVaR at α=0.1 (lower is better)"),
    ]:
        for i, shield in enumerate(shields):
            vals = [eval_data[(env, shield)][metric_idx] for env in envs]
            style = SHIELD_STYLES[shield]
            bars = ax.bar(
                x + offset[i] * width, vals, width,
                color=style["color"],
                label=style["label"],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(env_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axhline(0.05 if metric_idx == 1 else 0,
                   color="red", lw=0.8, ls=":", alpha=0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_summary_bars.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


# Entry point

if __name__ == "__main__":
    print("Generating plots...")
    plot_lambda()
    plot_cost()
    plot_intervention()
    plot_summary_bars()
    print(f"All plots saved to {OUT_DIR}/")
