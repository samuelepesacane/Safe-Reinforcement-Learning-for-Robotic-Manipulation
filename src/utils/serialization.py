from typing import Dict, Any
import os
import json


def save_config(config: Dict[str, Any], save_dir: str, filename: str = "config.json") -> str:
    """
    Save a configuration dictionary as a JSON file.

    The directory is created if it does not exist. The configuration is written
    with indentation to make inspection easier.

    :param config: Configuration mapping (e.g. hyperparameters, environment id).
        :type config: Dict[str, Any]
    :param save_dir: Directory where the JSON file will be stored.
        :type save_dir: str
    :param filename: Name of the JSON file to create inside ``save_dir``.
        :type filename: str

    :return: Absolute path to the written JSON configuration file.
        :rtype: str
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration dictionary from a JSON file.

    :param path: Path to a JSON file previously created with function `save_config`
        or an equivalent structure.
        :type path: str

    :return: Parsed configuration mapping.
        :rtype: Dict[str, Any]
    """
    with open(path, "r") as f:
        return json.load(f)
