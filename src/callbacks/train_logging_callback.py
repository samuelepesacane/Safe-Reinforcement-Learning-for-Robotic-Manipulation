from typing import Any
from stable_baselines3.common.callbacks import BaseCallback


class TrainLoggingCallback(BaseCallback):
    """
    Callback that periodically forwards Stable-Baselines3 training metrics
    (keys starting with "train/") to a custom project logger.

    This is intended to keep the project-level metrics store (e.g. JSON-lines
    + metrics.csv) synchronized with SB3's internal logger, so that
    train/.* metrics appear alongside eval/.* and lagrangian/.*
    when analyzing results.

    :param custom_logger: Project-level logger object that exposes a
        ``log_scalars(dict, step=int)`` method. Typically this is the
        repository's JSON-lines logger used for experiment tracking.
        :type custom_logger: Any
    :param log_freq: Frequency, in environment steps, at which to forward
        training metrics. When num_timesteps % log_freq == 0, the callback
        extracts all keys starting with "train/" from SB3's logger and
        passes them to custom_logger.
        :type log_freq: int
    :param verbose: Verbosity level passed to BaseCallback.
        :type verbose: int
    """

    def __init__(self, custom_logger: Any, log_freq: int = 5000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.custom_logger: Any = custom_logger
        self.log_freq: int = log_freq

    def _on_step(self) -> bool:
        """
        Hook called by Stable-Baselines3 at every environment step.

        When the total number of environment steps num_timesteps is a
        multiple of log_freq, this method:

        1. Reads the latest scalar values from SB3's logger
           (self.logger.name_to_value).
        2. Filters for keys starting with "train/".
        3. Forwards these key-value pairs to the custom logger via
           custom_logger.log_scalars(train_metrics, step=num_timesteps).

        This callback does **not** modify the learning process; it is purely
        for logging and experiment tracking.

        :return: Flag indicating whether training should continue. Always
            returns True.
            :rtype: bool
        """
        # Trigger only every `log_freq` environment steps
        if self.num_timesteps % self.log_freq == 0:
            # SB3 stores the most recent scalars in `self.logger.name_to_value`
            log_dict = getattr(self.logger, "name_to_value", {})
            train_metrics = {
                key: value
                for key, value in log_dict.items()
                if isinstance(key, str) and key.startswith("train/")
            }

            if train_metrics and self.custom_logger is not None:
                try:
                    self.custom_logger.log_scalars(
                        train_metrics,
                        step=self.num_timesteps,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"[TrainLoggingCallback] custom_logger failed: {exc}")

        return True
