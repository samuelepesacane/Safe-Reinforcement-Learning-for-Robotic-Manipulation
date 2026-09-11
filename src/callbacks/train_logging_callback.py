from typing import Any, Dict, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainLoggingCallback(BaseCallback):
    """
    Periodically forward training and episode-level safety metrics to a custom logger.

    Two sources of metrics are collected:

    1. SB3 train/* stats (loss, entropy, KL divergence, etc.), read from SB3's
       internal logger and forwarded as-is.
    2. Episode-level metrics (return, cost, violation rate, shield intervention
       rate and deflection magnitude), accumulated from infos at every step and
       averaged over the logging window. Intervention rate and deflection
       magnitude are logged together because the rate alone can't distinguish
       "intervenes often but gently" from "intervenes rarely but hard". For
       RiemannianShield, the gradient-stage-only magnitude, intervention rate,
       and (conditional) clip-fire rate are logged separately from the total
       deflection magnitude, which also includes any bisection fallback --
       averaging the two together previously hid that the gradient stage is
       bounded by alpha*max_action_norm and its norm clip can saturate on
       nearly every intervention.

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
        self._ep_deflections: List[float] = []
        self._ep_gradient_deflections: List[float] = []
        self._ep_gradient_interventions: List[int] = []
        self._ep_gradient_clip_fired: List[int] = []
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

            # Track per-step deflection magnitude alongside the intervention
            # rate: the rate alone can't distinguish "intervenes often but
            # gently" from "intervenes rarely but hard", which matters for
            # comparing a holonomic point robot against a nonholonomic car.
            # This is the TOTAL action change (gradient stage + any bisection
            # fallback for RiemannianShield); see the gradient-only fields
            # below for the mechanism this shield is actually characterized by.
            self._ep_deflections.append(float(info.get("shield_deflection_magnitude", 0.0)))

            # Gradient-stage-only diagnostics (RiemannianShield only; absent
            # keys default to 0/False for shields with no gradient stage).
            # Kept separate from the total above because conflating them hid
            # that the gradient-stage magnitude is bounded by
            # alpha*max_action_norm and the norm clip may saturate almost
            # every intervention -- see shield_gradient_clip_fire_rate.
            self._ep_gradient_deflections.append(
                float(info.get("shield_gradient_deflection_magnitude", 0.0))
            )
            gradient_intervened = bool(info.get("shield_gradient_intervened", False))
            self._ep_gradient_interventions.append(1 if gradient_intervened else 0)
            if gradient_intervened:
                self._ep_gradient_clip_fired.append(
                    1 if info.get("shield_gradient_clip_fired", False) else 0
                )

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

            if self._ep_deflections:
                metrics["train/shield_deflection_magnitude"] = float(
                    np.mean(self._ep_deflections)
                )

            if self._ep_gradient_deflections:
                metrics["train/shield_gradient_deflection_magnitude"] = float(
                    np.mean(self._ep_gradient_deflections)
                )
            if self._ep_gradient_interventions:
                metrics["train/shield_gradient_intervention_rate"] = float(
                    np.mean(self._ep_gradient_interventions)
                )
            if self._ep_gradient_clip_fired:
                # Conditional on a gradient intervention having happened this
                # step (see the accumulation above), not diluted by steps with
                # no gradient intervention at all.
                metrics["train/shield_gradient_clip_fire_rate"] = float(
                    np.mean(self._ep_gradient_clip_fired)
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
            self._ep_deflections.clear()
            self._ep_gradient_deflections.clear()
            self._ep_gradient_interventions.clear()
            self._ep_gradient_clip_fired.clear()
            self._step_costs.clear()

        return True
