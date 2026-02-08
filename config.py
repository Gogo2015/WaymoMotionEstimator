"""
Configuration management for experiments.
Loads YAML config files and provides easy access to parameters.
"""
import yaml
import os
import json
from datetime import datetime


class Config:
    """Configuration container with dot notation access."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """Get attribute with default fallback."""
        return getattr(self, key, default)

    def to_dict(self):
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with loaded parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config, save_dir):
    """
    Save configuration to experiment directory.

    Args:
        config: Config object
        save_dir: Directory to save config
    """
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Also save as JSON for easy reading
    json_path = os.path.join(save_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
