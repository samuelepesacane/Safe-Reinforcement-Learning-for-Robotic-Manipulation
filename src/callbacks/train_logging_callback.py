from typing import Any, Dict, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainLoggingCallback(BaseCallback):
    """
    Periodically forward training and episode-level safety metrics to a custom logger.

    Two sources of metrics are collected:

    1. SB3 train/* stats (loss, entropy, KL divergence, etc.), read from SB3's
       internal logger and forwarded as-is.
    2. Episode-level metrics (return, cost, violation rate, shield interventions),
       accumulated from infos at every step and averaged over the logging window.

    Averaging over the window ensures that PPO, SAC, RCPO, and LagPPO all produce
    comparable cost and return curves regardless of whether a Lagrangian callback
    is also present.

    :param custom_logger: Project-level logger exposing log_scalars(dict, step=int).
        :type custom_logger: Any
    :param log_freq: Frequency in environment steps at which metrics are flushed.
        :type log_freq: int
    :param cost_budget: Per-step cost budget used to compute violation rate.
        A step is a violation if its cost exceeds this value.
        :type cost_budget: float
    :param verbose: Verbosity level passed to BaseCallback.
        :type verbose: int
    """

    def __init__(
        self,
        custom_logger: Any,
        log_freq: int = 5000,
        cost_budget: float = 0.05,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.custom_logger: Any = custom_logger
        self.log_freq: int = log_freq
        self.cost_budget: float = cost_budget

        # Accumulators are reset every log_freq steps to avoid stale averages
        self._ep_returns: List[float] = []
        self._ep_costs: List[float] = []
        self._ep_interventions: List[int] = []
        self._step_costs: List[float] = []

    def _on_step(self) -> bool:
        """
        Accumulate per-step and per-episode metrics, then flush every log_freq steps.

        Episode-level stats (return, cost) are only available at episode end.
        SB3 wraps them under the "episode" key in info when the episode closes,
        so we check for that key rather than tracking them manually.

        :return: Always True (training continues).
            :rtype: bool
        """
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])

        for info in infos:
            step_cost = float(info.get("cost", 0.0))
            self._step_costs.append(step_cost)

            # Track per-step shield interventions so we can compute a rate
            self._ep_interventions.append(1 if info.get("shield_intervened", False) else 0)

            # Episode-level stats are only available when an episode ends
            ep_info = info.get("episode", None)
            if ep_info is not None:
                self._ep_returns.append(float(ep_info.get("r", 0.0)))
                self._ep_costs.append(float(ep_info.get("cost", 0.0)))

        if self.num_timesteps % self.log_freq == 0:
            metrics: Dict[str, float] = {}

            # Pull SB3 train/* stats directly from the internal logger
            log_dict = getattr(self.logger, "name_to_value", {})
            for key, value in log_dict.items():
                if isinstance(key, str) and key.startswith("train/"):
                    metrics[key] = float(value)

            if self._ep_returns:
                metrics["train/ep_return"] = float(np.mean(self._ep_returns))
            if self._ep_costs:
                metrics["train/ep_cost"] = float(np.mean(self._ep_costs))

            if self._step_costs:
                avg_step_cost = float(np.mean(self._step_costs))
                metrics["train/avg_cost_per_step"] = avg_step_cost
                # Fraction of steps that exceeded the budget, not just average cost,
                # because a low mean can hide frequent small violations
                metrics["train/violation_rate"] = float(
                    np.mean([c > self.cost_budget for c in self._step_costs])
                )

            if self._ep_interventions:
                metrics["train/shield_intervention_rate"] = float(
                    np.mean(self._ep_interventions)
                )

            if metrics and self.custom_logger is not None:
                try:
                    self.custom_logger.log_scalars(metrics, step=self.num_timesteps)
                except Exception as exc:
                    if self.verbose:
                        print(f"[TrainLoggingCallback] custom_logger failed: {exc}")

            # Reset after flushing so the next window starts clean
            self._ep_returns.clear()
            self._ep_costs.clear()
            self._ep_interventions.clear()
            self._step_costs.clear()

        return True
