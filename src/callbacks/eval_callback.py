from typing import Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):
    """
    Periodic evaluation callback that logs reward and safety metrics.

    At a fixed frequency in environment steps, this callback:
    - Evaluates the current policy on a separate evaluation environment to
      obtain mean episode return.
    - Rolls out additional episodes to accumulate total safety cost per episode
      from info["cost"].
    - Logs averaged return and cost to the SB3 logger (TensorBoard / stdout)
      and optionally to a custom project logger.
    - Optionally updates a shared multiprocessing.Value with the latest
      Lagrange multiplier, used to monitor the lambda trajectory in LagPPO.

    :param eval_env: Evaluation environment with info["cost"] at each step.
        :type eval_env: Any
    :param eval_freq: Training steps between evaluations.
        :type eval_freq: int
    :param n_eval_episodes: Number of episodes to average over at eval time.
        :type n_eval_episodes: int
    :param deterministic: Whether to use deterministic actions at eval time.
        :type deterministic: bool
    :param render: Whether to render evaluation episodes.
        :type render: bool
    :param verbose: Verbosity level forwarded to BaseCallback.
        :type verbose: int
    :param custom_logger: Optional project-level logger with
        log_scalars(dict, step=int).
        :type custom_logger: Optional[Any]
    :param shared_lambda_value: Optional multiprocessing.Value updated with
        the current lambda if the model exposes model.lag_state.lam.
        :type shared_lambda_value: Optional[Any]
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 0,
        custom_logger: Optional[Any] = None,
        shared_lambda_value: Optional[Any] = None,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.custom_logger = custom_logger
        self.shared_lambda_value = shared_lambda_value

    def _on_step(self) -> bool:
        """
        Evaluate the policy and log safety metrics every eval_freq steps.

        We compute cost by explicit rollout rather than relying on SB3's
        evaluate_policy because SB3's helper does not aggregate info["cost"].

        :return: Always True (training continues).
            :rtype: bool
        """
        if self.eval_freq <= 0:
            return True

        if self.n_calls % self.eval_freq != 0:
            return True

        # Use SB3's helper for returns; it handles vec_env and deterministic mode
        ep_returns, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True,
            deterministic=self.deterministic,
            render=self.render,
        )

        # Explicit rollout to accumulate cost, which evaluate_policy does not track
        costs = []
        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_cost = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_cost += float(info.get("cost", 0.0))
                done = bool(terminated or truncated)
            costs.append(ep_cost)

        avg_return = float(np.mean(ep_returns))
        avg_cost   = float(np.mean(costs))

        if self.logger is not None:
            self.logger.record("eval/avg_return", avg_return)
            self.logger.record("eval/avg_cost", avg_cost)

        if self.custom_logger is not None:
            try:
                self.custom_logger.log_scalars(
                    {"eval/avg_return": avg_return, "eval/avg_cost": avg_cost},
                    step=self.num_timesteps,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[EvalCallback] custom_logger failed: {exc}")

        # Update shared lambda so external monitors can track the multiplier
        # without needing direct access to the model
        if self.shared_lambda_value is not None:
            try:
                lag_state = getattr(self.model, "lag_state", None)
                lam = getattr(lag_state, "lam", None)
                if lam is not None:
                    self.shared_lambda_value.value = float(lam)
            except Exception:
                # Never let a failed lambda update interrupt training
                pass

        return True
