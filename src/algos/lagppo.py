from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class LagrangianState:
    """
    State of the Lagrangian dual variable for constrained RL.

    This object maintains the Lagrange multiplier `lam` and the hyperparameters
    governing its update, according to a simple dual ascent rule:

        lam <- max(0, lam + lr_lambda * (avg_cost_per_step - cost_budget))

    where `avg_cost_per_step` is an empirical estimate of the per-step cost
    over a recent window, and `cost_budget` is the desired average constraint.

    :param lam: Initial value of the Lagrange multiplier lambda.
    :type lam: float
    :param lr_lambda: Learning rate for the dual ascent update.
    :type lr_lambda: float
    :param cost_budget: Target average per-step cost (constraint budget).
    :type cost_budget: float
    :param update_every: Number of environment steps between dual updates
        (approximately the rollout length in on-policy methods).
    :type update_every: int
    :param clip_lam_max: Upper bound used to clip the Lagrange multiplier.
    :type clip_lam_max: float
    """
    lam: float = 0.0
    lr_lambda: float = 5e-4
    cost_budget: float = 0.05  # average per-step budget
    update_every: int = 2048   # update cadence (approx rollout length)
    clip_lam_max: float = 1e6

    def update(self, avg_cost_per_step: float) -> float:
        """
        Perform one dual ascent update on the Lagrange multiplier.

        :param avg_cost_per_step: Empirical average cost per step over the
            last `update_every` steps.
        :type avg_cost_per_step: float

        :return: Updated value of lambda.
        :rtype: float
        """
        # Dual ascent: lambda <- max(0, lambda + lr * (avg_cost - budget))
        delta: float = self.lr_lambda * (avg_cost_per_step - self.cost_budget)
        new_lam: float = max(0.0, min(self.clip_lam_max, self.lam + delta))
        self.lam = new_lam
        return self.lam


class LagrangianCallback(BaseCallback):
    """
    Stable-Baselines3 callback that tracks safety cost and updates the
    Lagrange multiplier lambda periodically.

    It assumes that the environment exposes a scalar cost in `info["cost"]`.
    Over a window of `update_every` steps, it aggregates cost, computes the
    average per-step cost, and updates `LagrangianState.lam` using dual ascent.

    The callback can optionally:

    - Log the current lambda and average cost to:
        * SB3's logger (for TensorBoard / stdout)
        * a custom JSON-lines logger, via `custom_logger.log_scalars(...)`
    - Synchronize lambda with parallel workers via `shared_lambda_value`
      (a multiprocessing.Value attached externally).

    :param lag_state: Shared LagrangianState instance whose `lam` will be
        updated in-place by this callback.
    :type lag_state: LagrangianState
    :param verbose: Verbosity flag inherited from BaseCallback.
    :type verbose: int
    """

    def __init__(self, lag_state: LagrangianState, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.lag_state: LagrangianState = lag_state
        self._cost_sum: float = 0.0
        self._step_count: int = 0

    def _on_step(self) -> bool:
        """
        Called by Stable-Baselines3 at each environment step.

        It reads `infos` from `self.locals` (populated by SB3), accumulates
        cost, and periodically updates the Lagrange multiplier.

        :return: Whether training should continue.
        :rtype: bool
        """
        # SB3 places env infos in locals["infos"] (one dict per env).
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                self._cost_sum += float(info.get("cost", 0.0))
                self._step_count += 1

        # Perform a dual update when enough steps have accumulated.
        if self._step_count >= self.lag_state.update_every:
            avg_cost: float = self._cost_sum / max(1, self._step_count)
            lam: float = self.lag_state.update(avg_cost)

            # Log to SB3 logger (TensorBoard / stdout).
            if self.logger is not None:
                self.logger.record("lagrangian/avg_cost_per_step", avg_cost)
                self.logger.record("lagrangian/lambda", lam)

            # Optionally log to the project-level JSON-lines logger.
            custom_logger = getattr(self, "custom_logger", None)
            if custom_logger is not None:
                try:
                    custom_logger.log_scalars(
                        {
                            "lagrangian/avg_cost_per_step": float(avg_cost),
                            "lagrangian/lambda": float(lam),
                        },
                        step=self.num_timesteps,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"[LagrangianCallback] failed to write custom_logger: {exc}")

            # Optionally propagate lambda to SubprocVecEnv workers.
            shared_lambda_value = getattr(self, "shared_lambda_value", None)
            if shared_lambda_value is not None:
                try:
                    shared_lambda_value.value = float(self.lag_state.lam)
                except Exception:
                    # Platform differences can cause failures; ignore them.
                    pass

            # Reset counters for the next window.
            self._cost_sum = 0.0
            self._step_count = 0

        return True


def make_lagppo(
    policy: str,
    env: Any,
    lag_state: LagrangianState,
    tensorboard_log: Optional[str] = None,
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[PPO, LagrangianCallback]:
    """
    Create a PPO model together with a Lagrangian callback for constrained RL.

    The model uses standard PPO hyperparameters (similar to `make_ppo`), while
    the `LagrangianCallback` updates the shared `LagrangianState.lam` based on
    observed safety costs. Combined with a reward-shaping wrapper
    (e.g. r' = r - lambda * cost), this implements a CMDP-style LagPPO scheme.

    :param policy: Policy class name or identifier for SB3's PPO.
    :type policy: str
    :param env: Environment instance or VecEnv compatible with SB3.
    :type env: Any
    :param lag_state: Shared LagrangianState controlling the dual variable.
    :type lag_state: LagrangianState
    :param tensorboard_log: Optional path for TensorBoard logs.
    :type tensorboard_log: Optional[str]
    :param seed: Random seed for the PPO model.
    :type seed: int
    :param config: Optional dictionary of PPO hyperparameters that overrides
        the defaults defined in this function.
    :type config: Optional[Dict[str, Any]]

    :return: Tuple of (PPO model, LagrangianCallback).
    :rtype: Tuple[stable_baselines3.PPO, LagrangianCallback]
    """
    cfg: Dict[str, Any] = dict(
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        vf_coef=0.5,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
    if config is not None:
        cfg.update(config)

    model: PPO = PPO(policy, env, **cfg)
    callback = LagrangianCallback(lag_state)
    return model, callback
