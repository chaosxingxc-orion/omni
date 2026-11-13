"""Configuration-driven execution orchestration, independent of study runtimes."""

from .config import ConfigError, load_config

__all__ = ["ConfigError", "load_config"]
