# gbb/config/__init__.py

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .defaults import DEFAULT_CONFIG, make_default_config
from .schemas import Config

cfg = DEFAULT_CONFIG


def load_config(
    data_dir: str | Path | None = None,
) -> Config:
    """
    Return an independent GBB configuration.

    When data_dir is given, construct a new configuration using that root.
    Otherwise, return a deep copy of DEFAULT_CONFIG.
    """
    if data_dir is not None:
        return make_default_config(data_dir=data_dir)

    return deepcopy(DEFAULT_CONFIG)


__all__ = [
    "Config",
    "cfg",
    "DEFAULT_CONFIG",
    "make_default_config",
    "load_config",
]