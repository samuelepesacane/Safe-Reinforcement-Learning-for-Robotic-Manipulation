from typing import Optional, Dict, Any
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Simple experiment logger that writes metrics to both JSON-lines and TensorBoard.

    Each call to `log_scalars` appends one JSON object to ``metrics.jsonl`` of
    the form ``{"step": int, "metric/name": value, ...}``, and also writes the same
    scalar values to a TensorBoard ``SummaryWriter``. The JSON-lines format is
    robust to evolving metric names and makes offline analysis with tools like
    pandas straightforward.
    """

    def __init__(self, log_dir: str, tb_dir: Optional[str] = None) -> None:
        """
        Create a new logger that writes metrics under ``log_dir``.

        :param log_dir: Directory where logging artifacts (e.g. ``metrics.jsonl``)
            will be stored. The directory is created if it does not exist.
            :type log_dir: str
        :param tb_dir: Optional directory for TensorBoard logs. If ``None``, a
            subdirectory under ``runs/`` is created, using the basename of
            ``log_dir`` and a timestamp.
            :type tb_dir: Optional[str]

        :return: This constructor does not return anything; it initializes
            the logger and underlying file handles.
            :rtype: None
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # JSON-lines path (robust to changing fields)
        self.json_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._json_file: Optional[Any] = None  # lazily opened

        # TensorBoard directory (timestamped by default)
        self.tb_dir = tb_dir or os.path.join(
            "runs",
            os.path.basename(log_dir) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.tb_dir, exist_ok=True)
        self.tb = SummaryWriter(self.tb_dir)

    def _ensure_open(self) -> None:
        """
        Ensure the JSON-lines file is open for appending.

        This helper is called internally before writing any metrics. It is
        separated out so that the metrics file is only created if logging
        actually occurs.
        """
        if self._json_file is None:
            # Open in append, line-buffered mode
            self._json_file = open(self.json_path, "a", buffering=1)

    def log_scalars(self, scalars: Dict[str, Any], step: int) -> None:
        """
        Log a batch of scalar metrics at a given global step.

        The metrics are written as a single JSON object on one line of the
        ``metrics.jsonl`` file and also recorded to TensorBoard. Non-numeric
        values are stored as-is in the JSON, but are skipped for TensorBoard.

        :param scalars: Mapping from metric name to numeric (or JSON-serializable)
            value. Metric names can include slashes (e.g. ``"train/ep_return"``).
            :type scalars: Dict[str, Any]
        :param step: Global training step associated with these measurements
            (e.g. timesteps or gradient updates).
            :type step: int

        :return: This method does not return anything; it performs I/O side effects.
            :rtype: None
        """
        self._ensure_open()
        row: Dict[str, Any] = {"step": int(step)}

        for k, v in scalars.items():
            # Try to coerce to float; if that fails, store the raw value.
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v

        # Write as one JSON line
        self._json_file.write(json.dumps(row) + "\n")

        # Mirror scalars to TensorBoard, skipping non-numeric values
        for k, v in scalars.items():
            try:
                self.tb.add_scalar(k, float(v), step)
            except Exception:
                # Ignore failures for non-numeric values
                pass

    def close(self) -> None:
        """
        Flush and close all underlying logging resources.

        This should be called at the end of training or when logging is no
        longer needed, to ensure all buffered metrics are written to disk.

        :return: This method does not return anything; it closes file handles.
            :rtype: None
        """
        try:
            if self._json_file is not None:
                self._json_file.flush()
                self._json_file.close()
        except Exception:
            # Do not raise on close; logging should not crash the program.
            pass

        try:
            self.tb.flush()
            self.tb.close()
        except Exception:
            pass
