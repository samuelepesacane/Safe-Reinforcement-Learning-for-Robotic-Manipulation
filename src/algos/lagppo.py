from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class LagrangianState:
    """
    State of the Lagrangian dual variable for constrained RL.

    Maintains the Lagrange multiplier lam and the hyperparameters governing
    its update via a simple dual ascent rule:

        lam <- clip(lam + lr_lambda * (avg_cost_per_step - cost_budget), 0, clip_lam_max)

    where avg_cost_per_step is the empirical per-step cost over a recent window
    and cost_budget is the desired average constraint.

    :param lam: Initial value of the Lagrange multiplier.
        :type lam: float
    :param lr_lambda: Learning rate for the dual ascent update.
        :type lr_lambda: float
    :param cost_budget: Target average per-step cost (constraint budget).
        :type cost_budget: float
    :param update_every: Environment steps between dual updates, set to
        approximately the rollout length for on-policy methods.
        :type update_every: int
    :param clip_lam_max: Upper bound for clipping the Lagrange multiplier.
        :type clip_lam_max: float
    """
    lam: float = 0.0
    lr_lambda: float = 5e-4
    cost_budget: float = 0.05
    update_every: int = 2048
    clip_lam_max: float = 1e6

    def update(self, avg_cost_per_step: float) -> float:
        """
        Perform one dual ascent step on the Lagrange multiplier.

        :param avg_cost_per_step: Empirical average cost per step over the
            last update_every steps.
            :type avg_cost_per_step: float

        :return: Updated value of lambda.
            :rtype: float
        """
        # Dual ascent: increase lambda when cost exceeds budget, decrease otherwise
        delta: float = self.lr_lambda * (avg_cost_per_step - self.cost_budget)
        self.lam = max(0.0, min(self.clip_lam_max, self.lam + delta))
        return self.lam


class LagrangianCallback(BaseCallback):
    """
    SB3 callback that tracks safety cost and updates the Lagrange multiplier.

    Assumes the environment exposes a scalar cost in info["cost"]. Over a
    window of update_every steps, it aggregates cost, computes the average
    per-step cost, and updates LagrangianState.lam via dual ascent.

    Optionally logs lambda and average cost to the SB3 logger (TensorBoard /
    stdout), to a custom JSON-lines logger, and propagates lambda to
    SubprocVecEnv workers via a shared multiprocessing.Value.

    :param lag_state: Shared LagrangianState whose lam is updated in-place.
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
        Accumulate cost and update the multiplier every update_every steps.

        SB3 places env infos in locals["infos"] (one dict per parallel env),
        so we iterate over the list rather than reading a single info dict.

        :return: Always True (training continues).
            :rtype: bool
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                self._cost_sum += float(info.get("cost", 0.0))
                self._step_count += 1

        if self._step_count >= self.lag_state.update_every:
            avg_cost: float = self._cost_sum / max(1, self._step_count)
            lam: float = self.lag_state.update(avg_cost)

            if self.logger is not None:
                self.logger.record("lagrangian/avg_cost_per_step", avg_cost)
                self.logger.record("lagrangian/lambda", lam)

            # Log to the project-level JSON-lines logger if one was attached
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

            # Propagate lambda to SubprocVecEnv workers so the reward
            # shaping wrapper in each worker uses the updated multiplier
            shared_lambda_value = getattr(self, "shared_lambda_value", None)
            if shared_lambda_value is not None:
                try:
                    shared_lambda_value.value = float(self.lag_state.lam)
                except Exception:
                    # Platform differences can cause multiprocessing.Value failures
                    pass

            # Reset counters so the next window starts from zero
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
) -> Tuple[PPO, LagrangianCallback]:
    """
    Create a PPO model and a LagrangianCallback for constrained RL training.

    The model uses the same PPO hyperparameters as make_ppo so that
    comparisons between PPO and LagPPO are controlled. The LagrangianCallback
    updates the shared LagrangianState.lam based on observed costs. Combined
    with RewardShapingWrapper (r' = r - lambda * cost), this implements the
    CMDP-style LagPPO described in Section 3.2 of the paper.

    :param policy: Policy class name accepted by SB3's PPO (e.g. "MlpPolicy").
        :type policy: str
    :param env: Environment instance or VecEnv compatible with SB3.
        :type env: Any
    :param lag_state: Shared LagrangianState controlling the dual variable.
        :type lag_state: LagrangianState
    :param tensorboard_log: Optional path for TensorBoard logs.
        :type tensorboard_log: Optional[str]
    :param seed: Random seed for the PPO model.
        :type seed: int
    :param config: Optional PPO hyperparameters that override the defaults.
        :type config: Optional[Dict[str, Any]]

    :return: Tuple of (PPO model, LagrangianCallback).
        :rtype: Tuple[PPO, LagrangianCallback]
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
