from typing import Dict
import yaml


def load_yaml(f_path: str) -> Dict:
    """

    Args:
        f_path: path to a yaml file.

    Returns: the loaded yaml file as a dict.

    """

    assert f_path.endswith(".yaml"), f"file path: {f_path} is not a yaml file path."

    with open(f_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config
