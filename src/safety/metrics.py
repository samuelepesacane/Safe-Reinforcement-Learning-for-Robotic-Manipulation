from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def cvar(values: List[float], alpha: float = 0.1) -> float:
    """
    Compute the Conditional Value at Risk (CVaR) at level alpha.

    CVaR_alpha is defined as the expected value of the worst alpha-fraction
    of outcomes. For costs, this measures the average tail risk.

    :param values: List of scalar values (e.g. episodic costs).
    :type values: List[float]
    :param alpha: Tail fraction in (0, 1]. For alpha=0.1, CVaR is the mean of
        the worst 10% of values.
    :type alpha: float, optional

    :return: CVaR at level alpha. Returns 0.0 if `values` is empty.
    :rtype: float
    """
    if len(values) == 0:
        return 0.0

    arr = np.array(values, dtype=np.float32)
    # Quantile threshold for the worst alpha fraction (upper tail).
    q = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr >= q]

    if len(tail) == 0:
        return float(q)
    return float(np.mean(tail))


def aggregate_episode_metrics(
    episodes: List[Dict[str, Any]],
    constraint_budget: Optional[float] = None,
) -> Dict[str, float]:
    """
    Aggregate per-episode results into summary safety and performance metrics.

    Each episode dictionary is expected to contain the following keys:
    - "returns": episodic return (float)
    - "cost": episodic safety cost (float)
    - "length": number of steps in the episode (int)
    - "interventions": number of shield interventions (int)
    - "success": whether the episode achieved the task goal (bool)

    The function computes:
    - avg_return: mean episodic return
    - avg_cost: mean episodic cost
    - avg_len: mean episode length
    - violation_rate: fraction of episodes with cost > 0
    - per_step_cost: total cost divided by total number of steps
    - cvar_cost_alpha_0.1: CVaR at alpha=0.1 over episodic costs
    - avg_interventions: mean number of interventions per episode
    - success_rate: fraction of successful episodes
    - return_under_budget: mean return among episodes whose average cost per
      step is <= constraint_budget (if a budget is provided)

    :param episodes: List of episode-level summaries.
    :type episodes: List[Dict[str, Any]]
    :param constraint_budget: Optional cost budget per step. If provided,
        `return_under_budget` is computed over episodes whose average
        per-step cost (cost / length) is within this budget.
    :type constraint_budget: Optional[float]

    :return: Dictionary mapping metric names to scalar values.
    :rtype: Dict[str, float]
    """
    returns = [float(ep.get("returns", 0.0)) for ep in episodes]
    costs = [float(ep.get("cost", 0.0)) for ep in episodes]
    lengths = [int(ep.get("length", 0)) for ep in episodes]
    successes = [bool(ep.get("success", False)) for ep in episodes]
    interventions = [int(ep.get("interventions", 0)) for ep in episodes]

    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_cost = float(np.mean(costs)) if costs else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0

    # Fraction of episodes with any safety violation.
    viol_rate = (
        float(np.mean([1.0 if c > 0.0 else 0.0 for c in costs]))
        if costs
        else 0.0
    )

    # Global per-step cost across all episodes.
    total_steps = int(np.sum(lengths)) if lengths else 0
    per_step_cost = float(np.sum(costs) / max(1, total_steps)) if costs else 0.0

    cvar_cost = cvar(costs, alpha=0.1)

    avg_interventions = float(np.mean(interventions)) if interventions else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0

    # Mean return conditioned on respecting a cost budget per step.
    ret_under_budget = 0.0
    if constraint_budget is not None and constraint_budget >= 0.0:
        qualified_returns: List[float] = []
        for r, c, l in zip(returns, costs, lengths):
            if l <= 0:
                continue
            avg_cost_per_step = c / float(l)
            if avg_cost_per_step <= constraint_budget:
                qualified_returns.append(r)
        ret_under_budget = (
            float(np.mean(qualified_returns)) if qualified_returns else 0.0
        )

    return {
        "avg_return": avg_return,
        "avg_cost": avg_cost,
        "avg_len": avg_len,
        "violation_rate": viol_rate,
        "per_step_cost": per_step_cost,
        "cvar_cost_alpha_0.1": cvar_cost,
        "avg_interventions": avg_interventions,
        "success_rate": success_rate,
        "return_under_budget": ret_under_budget,
    }


def dump_metrics_csv(metrics: Dict[str, float], path: str) -> None:
    """
    Save aggregated metrics to a single-row CSV file.

    :param metrics: Dictionary of metric name -> scalar value.
    :type metrics: Dict[str, float]
    :param path: Output CSV path.
    :type path: str

    :return: None
    :rtype: None
    """
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def log_to_tensorboard(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
) -> None:
    """
    Log evaluation metrics to TensorBoard.

    Each metric is logged under the tag "eval/<metric_name>".

    :param writer: TensorBoard SummaryWriter instance.
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param metrics: Dictionary of metric name -> scalar value.
    :type metrics: Dict[str, float]
    :param step: Global step at which to log the metrics.
    :type step: int

    :return: None
    :rtype: None
    """
    for k, v in metrics.items():
        writer.add_scalar(f"eval/{k}", v, step)
