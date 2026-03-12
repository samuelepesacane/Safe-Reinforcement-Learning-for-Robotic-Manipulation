from typing import Optional, Dict, Any
from stable_baselines3 import PPO


def make_ppo(
    policy: str,
    env: Any,
    tensorboard_log: Optional[str] = None,
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> PPO:
    """
    Create a PPO model with sensible defaults for continuous-control environments.

    This is a convenience factory used by train.py. Default hyperparameters
    can be overridden via the config dictionary.

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

    :return: Instantiated PPO model ready for training.
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
