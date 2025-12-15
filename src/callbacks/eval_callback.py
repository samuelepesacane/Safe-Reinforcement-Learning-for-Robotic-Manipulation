from typing import Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):
    """
    Periodic evaluation callback that logs reward and safety metrics.

    At a fixed frequency in terms of environment steps, this callback:
    - Evaluates the current policy on a separate evaluation environment using
      `stable_baselines3.common.evaluation.evaluate_policy` to obtain
      episode returns.
    - Rolls out additional evaluation episodes to accumulate total safety cost
      per episode (from ``info["cost"]``).
    - Logs averaged return and cost to the SB3 logger (for TensorBoard / stdout).
    - Optionally logs the same metrics to a custom project logger (e.g. JSON-lines).
    - Optionally updates a shared multiprocessing.Value with the latest
      Lagrange multiplier, if present on the model as ``model.lag_state.lam``.
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
        """
        Initialize the evaluation callback.

        :param eval_env: Evaluation environment compatible with the Gym/Gymnasium API.
            It must implement ``reset()`` and ``step(action)`` and provide safety
            costs in the info dict under the key ``"cost"``.
            :type eval_env: Any
        :param eval_freq: Number of training steps between two evaluation phases.
            Set to a non-positive value to disable evaluation.
            :type eval_freq: int
        :param n_eval_episodes: Number of episodes to average over at evaluation time.
            :type n_eval_episodes: int
        :param deterministic: Whether to use deterministic actions at evaluation time.
            :type deterministic: bool
        :param render: Whether to render the evaluation episodes.
            :type render: bool
        :param verbose: Verbosity level forwarded to BaseCallback.
            :type verbose: int
        :param custom_logger: Optional project-level logger with a method
            ``log_scalars(dict, step=int)`` for additional logging (e.g. JSON-lines).
            :type custom_logger: Optional[Any]
        :param shared_lambda_value: Optional ``multiprocessing.Value`` that is updated
            with the current Lagrange multiplier λ if available on the model.
            :type shared_lambda_value: Optional[Any]
        """
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
        Hook called by Stable-Baselines3 at every environment step.

        When the number of calls reaches a multiple of ``eval_freq``, this method:
        - Evaluates the current policy on ``n_eval_episodes`` to compute the mean
          return.
        - Computes mean total safety cost over ``n_eval_episodes`` by explicit
          rollouts using ``info["cost"]``.
        - Logs the resulting metrics.
        - Optionally updates a shared λ value for external monitoring.

        :return: Whether training should continue.
            :rtype: bool
        """
        if self.eval_freq <= 0:
            return True

        if self.n_calls % self.eval_freq != 0:
            return True

        # 1) Evaluate episodic returns using SB3 helper
        ep_returns, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True,
            deterministic=self.deterministic,
            render=self.render,
        )

        # 2) Compute total safety costs per episode by explicit rollouts
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
        avg_cost = float(np.mean(costs))

        # 3) Log to SB3 logger (TensorBoard / stdout)
        if self.logger is not None:
            self.logger.record("eval/avg_return", avg_return)
            self.logger.record("eval/avg_cost", avg_cost)

        # 4) Log to custom project logger if provided
        if self.custom_logger is not None:
            try:
                self.custom_logger.log_scalars(
                    {
                        "eval/avg_return": avg_return,
                        "eval/avg_cost": avg_cost,
                    },
                    step=self.num_timesteps,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[EvalCallback] custom_logger failed: {exc}")

        # 5) Optionally update shared λ (for LagPPO / constrained methods)
        if self.shared_lambda_value is not None:
            try:
                lag_state = getattr(self.model, "lag_state", None)
                lam = getattr(lag_state, "lam", None)
                if lam is not None:
                    self.shared_lambda_value.value = float(lam)
            except Exception:
                # Failing to update shared lambda should never interrupt training
                pass

        return True
