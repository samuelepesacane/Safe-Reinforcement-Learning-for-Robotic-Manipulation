from typing import Any, Tuple
import gymnasium as gym
import numpy as np
import torch
from .preference_model import PreferenceReward, flatten_obs


class RewardReplacementWrapper(gym.RewardWrapper):
    """
    Gymnasium RewardWrapper that replaces the environment reward with a
    learned preference-based reward r_hat(s)`.

    The wrapped environment is still stepped as usual, but at each step:

      1. The current observation is flattened and passed through the
         class`PreferenceReward` model.
      2. The scalar output is used as the new reward.
      3. The original environment reward is stored in ``info["env_reward"]``.

    This allows training agents on a learned reward model while retaining
    access to the ground-truth environment reward for evaluation.

    :param env: Underlying Gymnasium environment to wrap.
    :type env: gym.Env
    :param model: Trained reward model that maps observations to scalar rewards.
    :type model: PreferenceReward
    :param device: Device on which to run the reward model ("cpu" or "cuda").
    :type device: str
    """

    def __init__(self, env: gym.Env, model: PreferenceReward, device: str = "cpu") -> None:
        super().__init__(env)
        self.model = model.to(device)
        self.device = device

    def reward(self, reward: float) -> float:
        """
        Placeholder override required by gym.RewardWrapper.

        The actual reward replacement is done in `step`, since the
        default RewardWrapper API does not provide access to the observation
        when computing the reward.

        :param reward: Original environment reward (unused here).
        :type reward: float

        :return: The unchanged reward (ignored in practice).
        :rtype: float
        """
        return reward

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """
        Step the environment and replace the reward with the learned reward.

        The method:
          - calls ``env.step(action)`` to obtain (obs, reward, terminated, truncated, info),
          - computes ``r_hat = model(flatten_obs(obs))``,
          - stores the original environment reward in ``info["env_reward"]``,
          - returns the same (obs, terminated, truncated, info) but with the
            reward set to ``r_hat``.

        :param action: Action passed to the underlying environment.
        :type action: Any

        :return: Tuple (obs, reward_hat, terminated, truncated, info) where
            ``reward_hat`` is the preference-based reward.
        :rtype: Tuple[Any, float, bool, bool, dict]
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs_vec = flatten_obs(obs)
        with torch.no_grad():
            x = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)
            r_hat = float(self.model(x).item())

        # Preserve the environment's original reward for diagnostics
        info = dict(info) if info is not None else {}
        info["env_reward"] = float(reward)

        return obs, r_hat, terminated, truncated, info
