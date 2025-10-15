"""Utility helpers for loading and validating YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

import yaml


class ConfigError(RuntimeError):
    """Raised when the configuration file is malformed."""


def load_config(config_path: str | Path) -> MutableMapping[str, Any]:
    """Load a YAML configuration file and return it as a mutable mapping.

    The file path is resolved relative to the project root, ensuring that any
    nested relative paths declared inside the YAML remain anchored to the
    `master` directory structure.
    """

    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        raise ConfigError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, MutableMapping):
        raise ConfigError("Configuration root must be a mapping.")

    config["__config_file__"] = resolved_path
    config["__project_root__"] = resolved_path.parent.parent
    return config


def resolve_project_path(config: MutableMapping[str, Any], relative_path: str | Path) -> Path:
    """Resolve a project-relative path declared inside the configuration."""

    project_root = Path(config.get("__project_root__"))
    return (project_root / Path(relative_path)).resolve()
