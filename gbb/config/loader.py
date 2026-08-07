"""TOML configuration loading for GBB."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .defaults import make_default_config
from .schemas import Config


def _coerce_value(
    current: Any,
    value: Any,
    path: str,
) -> Any:
    """Coerce TOML containers to the types used by the dataclass defaults."""

    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"{path} must be bool, got {type(value).__name__}"
            )
        return value

    if isinstance(current, int) and not isinstance(current, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{path} must be int, got {type(value).__name__}"
            )
        return value

    if isinstance(current, float):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{path} must be numeric, got "
                f"{type(value).__name__}"
            )
        return float(value)

    if isinstance(current, str):
        if not isinstance(value, str):
            raise TypeError(
                f"{path} must be str, got {type(value).__name__}"
            )
        return value

    if isinstance(current, tuple):
        if not isinstance(value, list):
            raise TypeError(
                f"{path} must be a TOML array"
            )
        return tuple(value)

    if isinstance(current, list):
        if not isinstance(value, list):
            raise TypeError(
                f"{path} must be a TOML array"
            )
        return list(value)

    if current is None:
        return value

    return value


def _apply_mapping(
    target: Any,
    values: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    """Apply a TOML mapping recursively to a dataclass."""

    valid_fields = {item.name for item in fields(target)}
    unknown = sorted(set(values) - valid_fields)

    if unknown:
        location = prefix or "configuration"
        raise KeyError(
            f"Unknown {location} option(s): {unknown}"
        )

    for key, value in values.items():
        current = getattr(target, key)
        path = f"{prefix}.{key}" if prefix else key

        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(
                    f"{path} must be a TOML table"
                )

            _apply_mapping(
                current,
                value,
                prefix=path,
            )
            continue

        setattr(
            target,
            key,
            _coerce_value(current, value, path),
        )


def load_toml_config(
    path: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> Config:
    """Load a GBB configuration profile from TOML."""

    config_path = Path(path).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("rb") as handle:
        values = tomllib.load(handle)

    if "paths" in values:
        raise ValueError(
            "Do not configure machine-specific paths in the TOML "
            "profile. Set GBB_DATA_DIR instead."
        )

    configuration = make_default_config(
        data_dir=data_dir,
    )

    _apply_mapping(
        configuration,
        values,
    )

    return configuration.finalize()
