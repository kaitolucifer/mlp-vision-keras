from types import SimpleNamespace
import os

import yaml


def load_config() -> SimpleNamespace:
    file_path = os.path.dirname(__file__)
    config_path = os.path.join(file_path, "config.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)
