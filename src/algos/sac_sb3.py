from typing import Optional, Dict, Any
from stable_baselines3 import SAC


def make_sac(
    policy: str,
    env: Any,
    tensorboard_log: Optional[str] = None,
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> SAC:
    """
    Create a Stable-Baselines3 SAC model with defaults tailored to
    continuous-control tasks.

    The configuration is intended to serve as a strong off-policy baseline.
    Hyperparameters can be overridden via `config`.

    :param policy: Policy class name or identifier accepted by SB3's SAC
        (e.g. "MlpPolicy").
    :type policy: str
    :param env: Environment instance or VecEnv compatible with SB3.
    :type env: Any
    :param tensorboard_log: Optional path for TensorBoard logs.
    :type tensorboard_log: Optional[str]
    :param seed: Random seed for the SAC model.
    :type seed: int
    :param config: Optional dictionary of SAC hyperparameters that overrides
        the defaults defined in this function.
    :type config: Optional[Dict[str, Any]]

    :return: Instantiated SAC model ready for training.
    :rtype: stable_baselines3.SAC
    """
    cfg: Dict[str, Any] = dict(
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_rate=3e-4,
        train_freq=(1, "step"),
        gradient_steps=1,
        learning_starts=10_000,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
    if config is not None:
        cfg.update(config)

    model: SAC = SAC(policy, env, **cfg)
    return model
