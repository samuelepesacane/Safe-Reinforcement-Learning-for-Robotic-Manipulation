"""
Classify Lagrange multiplier trajectories as settled, limit-cycling, or climbing.

This is the pre-registered read-out for the experimental program described in
CODE_AUDIT_AND_EXECUTION_PLAN.md. The statistics and their thresholds were frozen
on 2026-08-07, before any run in that program existed, and live in
settling_thresholds.json. The point of freezing them is that a threshold chosen
after seeing the outcome is not a measurement, so this module deliberately has no
way to tune anything from the command line.

Statistics, all computed over the final `window_fraction` of training steps:

  S1  normalized slope   slope * window_length / mean(lambda over window)
      Fractional change of lambda across the window. Normalizing by the mean is
      what makes runs comparable when lambda reaches wildly different magnitudes:
      a raw slope cannot distinguish 100 -> 110 from 1 -> 11.

  S2  deceleration ratio (lambda(T) - lambda(T/2)) / (lambda(T/2) - lambda(0))
      Whether lambda's growth is slowing. A trajectory heading for a plateau
      decelerates; one that is genuinely running away does not. On the labelled
      set this is the only statistic that separates CarGoal from PointGoal.

  S3  residual dispersion  std(residuals about the S1 line) / |mean(lambda)|
      Distinguishes a flat-but-noisy limit cycle from a flat-and-quiet plateau.

  S4  feasibility guard    mean cost over the window, and the fraction of windows
      above budget. If lambda climbs while realized cost stays above budget, the
      honest reading is that the constrained problem may be infeasible for this
      policy class at this budget, and the multiplier is behaving correctly.

  S5  dual authority       max(lambda) * per_step_cost / (avg_return / avg_len)
      The fraction of the per-step reward signal that the cost penalty actually
      represents. If this is far below 1 the dual loop is open: lambda cannot
      move the policy, so no settling classification means anything. Reward-scale
      terms come from the run's eval metrics.csv, which is the converged policy's
      reward scale. Runs below the threshold are reported VOID rather than
      classified.

Usage:
    python -m src.analysis.settling logs/dualgain_*
    python -m src.analysis.settling --labelled-set      # the 27 calibration runs
    python -m src.analysis.settling logs/foo --csv out.csv
"""

from typing import Any, Dict, List, Optional, Tuple
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "settling_thresholds.json")

# The 27 pre-existing LagPPO runs used to calibrate the frozen thresholds.
# Kept here so the calibration is reproducible by name, not by glob accident.
LABELLED_SET: List[str] = (
    [f"logs/car_lagppo_shield_{s}_seed{n}" for s in ("off", "on") for n in range(3)]
    + [f"logs/car_riemannian_lagppo_seed{n}" for n in range(3)]
    + [f"logs/goal_lagppo_shield_{s}_seed{n}" for s in ("off", "on") for n in range(3)]
    + [f"logs/goal_riemannian_lagppo_seed{n}" for n in range(3)]
    + [f"logs/lagppo_shield_{s}_seed{n}" for s in ("off", "on") for n in range(3)]
    + [f"logs/push_riemannian_lagppo_seed{n}" for n in range(3)]
)

# Ground-truth labels for the calibration set. PointPush is deliberately absent:
# it behaves inconsistently in both directions and must not drive thresholds.
LABELS: Dict[str, str] = {"car": "climbing", "goal": "settled"}


def load_thresholds(path: str = THRESHOLD_PATH) -> Dict[str, Any]:
    """
    Load the frozen threshold constants.

    :param path: Path to settling_thresholds.json.
        :type path: str

    :return: Parsed threshold dictionary.
        :rtype: Dict[str, Any]
    """
    with open(path) as f:
        return json.load(f)


def read_lambda_series(
    log_dir: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read the lambda and cost series from a run's metrics.jsonl.

    The file interleaves two cadences: LagrangianCallback rows carrying
    "lagrangian/lambda" every update_every=2048 env-steps summed across workers
    (492 rows in a 1M-step 4-worker run), and TrainLoggingCallback rows carrying
    only "train/*" keys every eval_freq steps (50 rows). We therefore filter on
    key presence and never on row index.

    :param log_dir: Directory containing metrics.jsonl.
        :type log_dir: str

    :return: (steps, lambda, avg_cost_per_step), or None if no lambda was logged.
        :rtype: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    path = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return None

    steps: List[float] = []
    lam: List[float] = []
    cost: List[float] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line from an interrupted run should not be fatal
                continue
            if "lagrangian/lambda" in row:
                steps.append(float(row["step"]))
                lam.append(float(row["lagrangian/lambda"]))
                cost.append(float(row.get("lagrangian/avg_cost_per_step", np.nan)))

    if len(lam) < 10:
        return None

    return np.asarray(steps), np.asarray(lam), np.asarray(cost)


def find_eval_metrics(log_dir: str) -> Optional[Dict[str, float]]:
    """
    Locate the eval metrics.csv matching a training log directory.

    Training logs are named logs/<tag> and their evaluations results/eval_<tag>,
    so the mapping is by convention rather than recorded anywhere.

    The original Push runs break that convention: they are logged as
    logs/lagppo_shield_off_seed0 but evaluated into
    results/eval_push_lagppo_shield_off_seed0. A naive suffix match is NOT a safe
    way to bridge that seam, because "car_lagppo_shield_off_seed0" also ends with
    "lagppo_shield_off_seed0" and would silently supply the wrong environment's
    reward scale. We therefore bridge it with an explicit rule -- a tag carrying
    no environment prefix is an original Push run -- and treat any residual
    ambiguity as an error rather than guessing.

    Returns None when no unambiguous evaluation exists, in which case S5 is NaN
    and the run is reported without an authority guard rather than with a wrong one.

    :param log_dir: Path to the training log directory.
        :type log_dir: str

    :return: Parsed single-row metrics.csv, or None.
        :rtype: Optional[Dict[str, float]]
    """
    tag = os.path.basename(os.path.normpath(log_dir))
    root = os.path.dirname(os.path.dirname(os.path.normpath(log_dir))) or "."

    # Tags from the original single-environment study carry no env prefix.
    known_prefixes = ("push_", "goal_", "car_", "dualgain_", "cripple_")
    names = [f"eval_{tag}"]
    if not tag.startswith(known_prefixes):
        names.append(f"eval_push_{tag}")

    for results_root in (os.path.join(root, "results"), "results"):
        found = [
            os.path.join(results_root, n, "metrics.csv")
            for n in names
            if os.path.isfile(os.path.join(results_root, n, "metrics.csv"))
        ]
        if len(found) > 1:
            print(f"[settling] ambiguous eval dirs for {tag}: {found} -- refusing to guess",
                  file=sys.stderr)
            return None
        if found:
            with open(found[0]) as f:
                rows = list(csv.DictReader(f))
            if rows:
                out: Dict[str, float] = {}
                for k, v in rows[0].items():
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                return out
    return None


def compute_statistics(
    log_dir: str,
    window_fraction: float,
    budget: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """
    Compute S1-S5 for one run.

    :param log_dir: Directory containing metrics.jsonl.
        :type log_dir: str
    :param window_fraction: Trailing fraction of training steps forming the window.
        :type window_fraction: float
    :param budget: Per-step cost budget, used for the S4 exceedance fraction.
        :type budget: float

    :return: Dictionary of statistics, or None if the run has no lambda series.
        :rtype: Optional[Dict[str, Any]]
    """
    series = read_lambda_series(log_dir)
    if series is None:
        return None
    steps, lam, cost = series

    # Trailing window
    cut = steps[-1] - window_fraction * (steps[-1] - steps[0])
    mask = steps >= cut
    sw, lw, cw = steps[mask], lam[mask], cost[mask]

    mean_lw = float(lw.mean())

    # S1: OLS line over the window, slope normalized by the window mean
    design = np.vstack([sw, np.ones_like(sw)]).T
    slope, intercept = np.linalg.lstsq(design, lw, rcond=None)[0]
    residuals = lw - (slope * sw + intercept)
    window_length = float(sw[-1] - sw[0])

    s1 = float(slope * window_length / mean_lw) if mean_lw != 0.0 else float("nan")
    s3 = float(residuals.std() / abs(mean_lw)) if mean_lw != 0.0 else float("nan")

    # S2: deceleration ratio across the two halves of the whole run
    half = lam[int(0.5 * (len(lam) - 1))]
    gain_first = float(half - lam[0])
    gain_second = float(lam[-1] - half)
    s2 = gain_second / gain_first if gain_first != 0.0 else float("nan")

    # S4: feasibility guard
    finite_cost = cw[np.isfinite(cw)]
    s4_cost = float(finite_cost.mean()) if finite_cost.size else float("nan")
    s4_over = float((finite_cost > budget).mean()) if finite_cost.size else float("nan")

    # S5: dual authority, using the converged policy's reward scale from eval
    ev = find_eval_metrics(log_dir)
    s5 = float("nan")
    if ev:
        avg_len = ev.get("avg_len", 0.0)
        avg_return = ev.get("avg_return", 0.0)
        per_step_cost = ev.get("per_step_cost", 0.0)
        if avg_len > 0 and avg_return != 0.0:
            reward_per_step = avg_return / avg_len
            s5 = abs(float(lam.max()) * per_step_cost / reward_per_step)

    return {
        "run": os.path.basename(os.path.normpath(log_dir)),
        "n_updates": int(len(lam)),
        "lam_max": float(lam.max()),
        "lam_final": float(lam[-1]),
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4_cost": s4_cost,
        "s4_frac_over_budget": s4_over,
        "s5": s5,
    }


def classify(stats: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    """
    Apply the frozen decision rule to one run's statistics.

    Order matters: the authority guard runs first, because a run whose multiplier
    cannot influence the policy has not demonstrated anything about settling and
    should not be given a settling label.

    :param stats: Output of compute_statistics.
        :type stats: Dict[str, Any]
    :param thresholds: Output of load_thresholds.
        :type thresholds: Dict[str, Any]

    :return: One of "VOID", "settled", "limit_cycle", "climbing", "undetermined".
        :rtype: str
    """
    s1, s2, s3, s5 = stats["s1"], stats["s2"], stats["s3"], stats["s5"]

    if np.isfinite(s5) and s5 < thresholds["authority_void"]["s5_min"]:
        return "VOID"

    if np.isfinite(s1) and np.isfinite(s3):
        if s1 <= thresholds["limit_cycle"]["s1_max"] and s3 >= thresholds["limit_cycle"]["s3_min"]:
            return "limit_cycle"

    if np.isfinite(s1) and np.isfinite(s2):
        if s1 <= thresholds["settled"]["s1_max"] and s2 <= thresholds["settled"]["s2_max"]:
            return "settled"

    if np.isfinite(s2) and s2 >= thresholds["climbing"]["s2_min"]:
        return "climbing"

    return "undetermined"


def main() -> None:
    """
    Report settling statistics for the given runs.

    Every seed is reported individually; the continuous statistics are the
    primary quantity and the classification is a secondary summary. With a
    handful of seeds a 4/5-vs-5/5 binary split carries almost no information,
    so the binary is never reported without the continuous values beside it.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dirs", nargs="*", help="Training log directories (globs allowed).")
    ap.add_argument("--labelled-set", action="store_true",
                    help="Use the 27 pre-existing LagPPO runs the thresholds were calibrated on.")
    ap.add_argument("--budget", type=float, default=0.05,
                    help="Per-step cost budget, for the S4 exceedance fraction.")
    ap.add_argument("--csv", type=str, default=None, help="Optional path to write the table as CSV.")
    args = ap.parse_args()

    thresholds = load_thresholds()

    if args.labelled_set:
        dirs = [d for d in LABELLED_SET if os.path.isdir(d)]
    else:
        dirs = []
        for pattern in args.log_dirs:
            dirs.extend(sorted(glob.glob(pattern)) or ([pattern] if os.path.isdir(pattern) else []))

    if not dirs:
        print("No log directories matched. Run from the repository root.", file=sys.stderr)
        raise SystemExit(1)

    print(f"thresholds frozen {thresholds['frozen_on']}  "
          f"window = final {thresholds['window_fraction']:.0%} of steps\n")
    header = (f"{'run':<38}{'S1':>8}{'S2':>8}{'S3':>8}{'S5':>8}"
              f"{'lam_max':>10}{'cost':>8}{'>bud':>7}  class")
    print(header)
    print("-" * len(header))

    rows: List[Dict[str, Any]] = []
    for d in dirs:
        stats = compute_statistics(d, thresholds["window_fraction"], budget=args.budget)
        if stats is None:
            print(f"{os.path.basename(os.path.normpath(d)):<38}{'  no lambda series logged':<60}")
            continue
        stats["class"] = classify(stats, thresholds)
        rows.append(stats)
        print(f"{stats['run']:<38}{stats['s1']:>8.3f}{stats['s2']:>8.3f}{stats['s3']:>8.3f}"
              f"{stats['s5']:>8.3f}{stats['lam_max']:>10.5f}{stats['s4_cost']:>8.4f}"
              f"{stats['s4_frac_over_budget']:>7.2f}  {stats['class']}")

    if rows:
        print()
        counts: Dict[str, int] = {}
        for r in rows:
            counts[r["class"]] = counts.get(r["class"], 0) + 1
        print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

        voids = [r for r in rows if r["class"] == "VOID"]
        if voids:
            print(f"\nWARNING: {len(voids)}/{len(rows)} runs are VOID -- the multiplier carries "
                  f"under {thresholds['authority_void']['s5_min']:.0%} of the per-step reward "
                  f"signal, so the dual loop is open and no settling claim is supportable "
                  f"for them.")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
