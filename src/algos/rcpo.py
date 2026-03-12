"""
Factory for the RCPO fixed-penalty baseline.

Not used in the current training pipeline. Planned for a future extension
that will compare against a faithful implementation of Tessler et al. (2018).
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from stable_baselines3 import PPO


@dataclass
class RCPOConfig:
    """
    Configuration for the RCPO fixed-penalty baseline.

    In this project, "RCPO" refers to a PPO agent trained on a
    penalty-shaped reward of the form:

        r' = r - penalty_coef * cost

    where cost is a scalar safety signal provided by the environment.
    The shaping itself is applied by the reward wrapper in src/envs;
    this dataclass only stores the coefficient used by that wrapper.

    :param penalty_coef: Fixed penalty coefficient multiplying the safety cost.
        :type penalty_coef: float
    """
    penalty_coef: float = 10.0


def make_rcpo(
    policy: str,
    env: Any,
    tensorboard_log: Optional[str] = None,
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> PPO:
    """
    Create a PPO model configured for RCPO-style constrained RL training.

    The PPO hyperparameters mirror those in make_ppo so that comparisons
    between plain PPO and RCPO are controlled. The only difference between
    the two in this codebase is the reward signal: for RCPO, the environment
    reward is externally reshaped as r' = r - penalty_coef * cost using a
    fixed coefficient (see RCPOConfig and the reward wrappers in src/envs).

    :param policy: Policy class name accepted by SB3's PPO (e.g. "MlpPolicy").
        :type policy: str
    :param env: Environment instance or VecEnv compatible with SB3.
        :type env: Any
    :param tensorboard_log: Optional path for TensorBoard logs.
        :type tensorboard_log: Optional[str]
    :param seed: Random seed for the PPO model.
        :type seed: int
    :param config: Optional PPO hyperparameters that override the defaults.
        :type config: Optional[Dict[str, Any]]

    :return: Instantiated PPO model configured for RCPO-style training.
        :rtype: PPO
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

    return PPO(policy, env, **cfg)
