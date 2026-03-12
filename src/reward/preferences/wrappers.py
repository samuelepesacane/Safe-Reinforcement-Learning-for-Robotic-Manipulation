"""
Gymnasium wrapper that replaces environment reward with a learned preference reward.

Supports the --use_preferences condition in the ablation grid. This condition
was explored during the study but is not included in the paper's main results.
"""
from typing import Any, Tuple
import gymnasium as gym
import numpy as np
import torch
from .preference_model import PreferenceReward, flatten_obs


class RewardReplacementWrapper(gym.RewardWrapper):
    """
    Replace the environment reward with a learned preference-based reward r_hat(s).

    At each step the current observation is flattened, passed through the
    PreferenceReward model, and the scalar output is used as the new reward.
    The original environment reward is preserved in info["env_reward"] so it
    remains accessible for evaluation without influencing training.

    :param env: Underlying Gymnasium environment to wrap.
        :type env: gym.Env
    :param model: Trained reward model that maps observations to scalar rewards.
        :type model: PreferenceReward
    :param device: Device on which to run the reward model.
        :type device: str
    """

    def __init__(self, env: gym.Env, model: PreferenceReward, device: str = "cpu") -> None:
        super().__init__(env)
        self.model = model.to(device)
        self.device = device

    def reward(self, reward: float) -> float:
        """
        Placeholder required by gym.RewardWrapper.

        The actual replacement is done in step() because the base class API
        does not pass the observation to this method, and we need it to compute
        r_hat(s).

        :param reward: Original environment reward (unused).
            :type reward: float

        :return: The reward unchanged.
            :rtype: float
        """
        return reward

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """
        Step the environment and replace the reward with the learned reward.

        :param action: Action passed to the underlying environment.
            :type action: Any

        :return: (obs, r_hat, terminated, truncated, info) where r_hat is the
            preference-based reward and info["env_reward"] holds the original.
            :rtype: Tuple[Any, float, bool, bool, dict]
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs_vec = flatten_obs(obs)
        with torch.no_grad():
            x = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)
            r_hat = float(self.model(x).item())

        # Keep the original reward for diagnostics without letting it influence training
        info = dict(info) if info is not None else {}
        info["env_reward"] = float(reward)

        return obs, r_hat, terminated, truncated, info
