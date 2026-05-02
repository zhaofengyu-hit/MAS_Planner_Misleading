"""Configuration loader utility."""

import yaml
from typing import Dict, Any, Optional
import os
import re
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration with environment variable support."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        self._config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def _expand_env_vars(self, value: Any) -> Any:
        """Recursively expand environment variables in configuration values.

        Supported formats:
        - ${VAR_NAME} - environment variable that must exist
        - ${VAR_NAME:default_value} - environment variable with a fallback default

        Args:
            value: Configuration value (may contain environment variable references).

        Returns:
            Value with environment variables expanded.
        """
        if isinstance(value, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default_value}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2)

                # Prefer the environment variable value
                env_value = os.environ.get(var_name)

                if env_value is not None:
                    return env_value
                elif default_value is not None:
                    return default_value
                else:
                    # No default and env var not set — raise an error
                    raise ValueError(
                        f"Environment variable '{var_name}' is required but not set"
                    )

            return re.sub(pattern, replacer, value)

        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]

        else:
            return value

    def load(self):
        """Load the configuration file and expand environment variables."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        with open(self._config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Expand environment variables
        self._config = self._expand_env_vars(raw_config)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})

    def update(self, key: str, value: Any):
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self):
        with open(self._config_path, "w") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        self.update(key, value)
