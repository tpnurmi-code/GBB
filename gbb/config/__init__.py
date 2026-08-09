# gbb/config/__init__.py

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from importlib import reload
from pathlib import Path

from .defaults import DEFAULT_CONFIG, make_default_config
from .loader import load_toml_config
from .schemas import Config

cfg = DEFAULT_CONFIG


def _copy_config_in_place(
    target,
    source,
) -> None:
    """Update dataclass values without invalidating imported references."""

    for item in fields(target):
        target_value = getattr(target, item.name)
        source_value = getattr(source, item.name)

        if (
            is_dataclass(target_value)
            and is_dataclass(source_value)
        ):
            _copy_config_in_place(
                target_value,
                source_value,
            )
        else:
            setattr(
                target,
                item.name,
                deepcopy(source_value),
            )


def load_config(
    config_path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> Config:
    """Return an independent GBB configuration."""

    if config_path is not None:
        return load_toml_config(
            config_path,
            data_dir=data_dir,
        )

    if data_dir is not None:
        return make_default_config(
            data_dir=data_dir,
        )

    return deepcopy(DEFAULT_CONFIG)


def activate_config(
    config_path: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> Config:
    """
    Load and activate a TOML configuration for the current process.

    Existing references to ``cfg`` remain valid. The flat compatibility
    module is reloaded so legacy ``config.X`` values match the new profile.
    """

    loaded = load_toml_config(
        config_path,
        data_dir=data_dir,
    )

    _copy_config_in_place(
        cfg,
        loaded,
    )

    cfg.finalize()

    # Update the transitional flat compatibility API.
    from . import config as flat_config

    reload(flat_config)

    return cfg


__all__ = [
    "Config",
    "cfg",
    "DEFAULT_CONFIG",
    "make_default_config",
    "load_config",
    "load_toml_config",
    "activate_config",
]
