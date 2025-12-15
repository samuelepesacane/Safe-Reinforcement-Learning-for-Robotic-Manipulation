from typing import Optional, Dict, Any
from dataclasses import dataclass
from stable_baselines3 import PPO


# This file is still incomplete. I am planning to finish it in the next version.

@dataclass
class RCPOConfig:
    """
    Configuration for an RCPO-style fixed-penalty baseline.

    In this project, "RCPO" refers to a PPO agent trained on a
    penalty-shaped reward of the form:

        r' = r - penalty_coef * cost

    where ``cost`` is a scalar safety signal provided by the environment.
    The actual shaping is implemented outside the model (e.g. in a reward
    wrapper); this dataclass simply stores the penalty coefficient used
    by that wrapper.

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
    Create a PPO model to be used in an RCPO-style constrained RL setup.

    The PPO hyperparameters used here mirror those in ``make_ppo`` so that
    comparisons between PPO and RCPO are controlled. The difference between
    PPO and RCPO in this repository lies in the *reward signal* used during
    training: for RCPO, the environment's reward is externally reshaped as

        r' = r - penalty_coef * cost

    using a fixed penalty coefficient (see RCPOConfig and the
    reward shaping wrappers in src/envs).

    :param policy: Policy class or string identifier for SB3's PPO (e.g. "MlpPolicy").
    :type policy: str
    :param env: Environment instance or VecEnv compatible with SB3.
    :type env: Any
    :param tensorboard_log: Optional path for TensorBoard logs.
    :type tensorboard_log: Optional[str]
    :param seed: Random seed for the PPO model.
    :type seed: int
    :param config: Optional dictionary of PPO hyperparameters. Any keys provided
        here override the defaults specified in this function.
    :type config: Optional[Dict[str, Any]]

    :return: Instantiated PPO model configured for RCPO-style training.
    :rtype: stable_baselines3.PPO
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
    return model
