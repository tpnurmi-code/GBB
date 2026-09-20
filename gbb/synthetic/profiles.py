"""Versioned ground-truth profiles for synthetic GBB benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .anatomy import SyntheticAnatomy

DistributionKind = Literal["fixed", "normal", "uniform"]

NODE_PARAMETER_NAMES = frozenset(
    {
        "tau_seconds",
        "intrinsic_drive",
        "stimulus_gain",
    }
)

NODE_SELECTOR_NAMES = frozenset(
    {
        "region",
        "hemisphere",
        "layer",
        "column_id",
        "network_id",
        "label",
    }
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: object, *, field_name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """One scalar ground-truth distribution."""

    distribution: DistributionKind

    value: float | None = None

    mean: float | None = None
    sd: float | None = None

    low: float | None = None
    high: float | None = None

    clip: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "DistributionSpec":
        allowed = {
            "distribution",
            "value",
            "mean",
            "sd",
            "low",
            "high",
            "clip",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"Unknown distribution fields: {sorted(unknown)}"
            )

        distribution = payload.get("distribution")
        if distribution not in {"fixed", "normal", "uniform"}:
            raise ValueError(
                "distribution must be one of: fixed, normal, uniform"
            )

        clip_payload = payload.get("clip")
        clip: tuple[float, float] | None = None

        if clip_payload is not None:
            if (
                not isinstance(clip_payload, list)
                or len(clip_payload) != 2
            ):
                raise ValueError("clip must be a two-element JSON list")

            clip_low = _as_float(
                clip_payload[0],
                field_name="clip[0]",
            )
            clip_high = _as_float(
                clip_payload[1],
                field_name="clip[1]",
            )

            if clip_high < clip_low:
                raise ValueError("clip upper bound must be >= lower bound")

            clip = (clip_low, clip_high)

        if distribution == "fixed":
            if "value" not in payload:
                raise ValueError(
                    "fixed distribution requires 'value'"
                )

            return cls(
                distribution="fixed",
                value=_as_float(
                    payload["value"],
                    field_name="value",
                ),
                clip=clip,
            )

        if distribution == "normal":
            if "mean" not in payload or "sd" not in payload:
                raise ValueError(
                    "normal distribution requires 'mean' and 'sd'"
                )

            mean = _as_float(
                payload["mean"],
                field_name="mean",
            )
            sd = _as_float(
                payload["sd"],
                field_name="sd",
            )

            if sd < 0:
                raise ValueError("normal distribution sd must be >= 0")

            return cls(
                distribution="normal",
                mean=mean,
                sd=sd,
                clip=clip,
            )

        if "low" not in payload or "high" not in payload:
            raise ValueError(
                "uniform distribution requires 'low' and 'high'"
            )

        low = _as_float(
            payload["low"],
            field_name="low",
        )
        high = _as_float(
            payload["high"],
            field_name="high",
        )

        if high < low:
            raise ValueError(
                "uniform distribution high must be >= low"
            )

        return cls(
            distribution="uniform",
            low=low,
            high=high,
            clip=clip,
        )

    def sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> np.ndarray:
        """Draw deterministic samples using the supplied RNG."""

        if size < 0:
            raise ValueError("size must be non-negative")

        if self.distribution == "fixed":
            assert self.value is not None
            values = np.full(
                size,
                self.value,
                dtype=np.float64,
            )

        elif self.distribution == "normal":
            assert self.mean is not None
            assert self.sd is not None

            values = rng.normal(
                self.mean,
                self.sd,
                size=size,
            )

        elif self.distribution == "uniform":
            assert self.low is not None
            assert self.high is not None

            values = rng.uniform(
                self.low,
                self.high,
                size=size,
            )

        else:  # pragma: no cover - guarded by validation
            raise RuntimeError(
                f"Unsupported distribution {self.distribution!r}"
            )

        if self.clip is not None:
            values = np.clip(
                values,
                self.clip[0],
                self.clip[1],
            )

        return np.asarray(values, dtype=np.float64)


def _validate_selector_value(
    selector: str,
    value: object,
) -> None:
    values = value if isinstance(value, list) else [value]

    if not values:
        raise ValueError(
            f"Selector {selector!r} cannot use an empty list"
        )

    integer_selector = selector in {
        "column_id",
        "network_id",
    }

    for item in values:
        if integer_selector:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(
                    f"Selector {selector!r} requires integer values"
                )
        else:
            if not isinstance(item, str):
                raise ValueError(
                    f"Selector {selector!r} requires string values"
                )


@dataclass(frozen=True, slots=True)
class NodeGroundTruthRule:
    """Override node parameters for nodes matching anatomical selectors."""

    name: str
    match: dict[str, object]
    parameters: dict[str, DistributionSpec]
    required: bool = True

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        index: int,
    ) -> "NodeGroundTruthRule":
        allowed = {
            "name",
            "match",
            "parameters",
            "required",
        }

        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"Unknown rule fields: {sorted(unknown)}"
            )

        name = str(
            payload.get(
                "name",
                f"rule_{index:03d}",
            )
        )

        match_payload = payload.get("match")
        if (
            not isinstance(match_payload, dict)
            or not match_payload
        ):
            raise ValueError(
                f"Rule {name!r} requires a non-empty 'match' object"
            )

        unknown_selectors = (
            set(match_payload) - NODE_SELECTOR_NAMES
        )
        if unknown_selectors:
            raise ValueError(
                f"Rule {name!r} contains unknown selectors: "
                f"{sorted(unknown_selectors)}"
            )

        for selector, selector_value in match_payload.items():
            _validate_selector_value(
                selector,
                selector_value,
            )

        parameter_payload = payload.get("parameters")
        if (
            not isinstance(parameter_payload, dict)
            or not parameter_payload
        ):
            raise ValueError(
                f"Rule {name!r} requires a non-empty "
                "'parameters' object"
            )

        unknown_parameters = (
            set(parameter_payload) - NODE_PARAMETER_NAMES
        )
        if unknown_parameters:
            raise ValueError(
                f"Rule {name!r} contains unsupported parameters: "
                f"{sorted(unknown_parameters)}"
            )

        parameters: dict[str, DistributionSpec] = {}

        for parameter_name, spec_payload in parameter_payload.items():
            if not isinstance(spec_payload, dict):
                raise ValueError(
                    f"Parameter {parameter_name!r} in rule "
                    f"{name!r} must be an object"
                )

            parameters[parameter_name] = (
                DistributionSpec.from_mapping(spec_payload)
            )

        required = payload.get("required", True)
        if not isinstance(required, bool):
            raise ValueError(
                f"Rule {name!r}: required must be true or false"
            )

        return cls(
            name=name,
            match=dict(match_payload),
            parameters=parameters,
            required=required,
        )


@dataclass(frozen=True, slots=True)
class NodeParameterProfile:
    """Default and ROI-specific node parameter definitions."""

    defaults: dict[str, DistributionSpec]
    rules: tuple[NodeGroundTruthRule, ...]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "NodeParameterProfile":
        allowed = {
            "defaults",
            "rules",
        }

        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"Unknown node_parameters fields: "
                f"{sorted(unknown)}"
            )

        defaults_payload = payload.get("defaults", {})
        if not isinstance(defaults_payload, dict):
            raise ValueError(
                "node_parameters.defaults must be an object"
            )

        unknown_defaults = (
            set(defaults_payload) - NODE_PARAMETER_NAMES
        )
        if unknown_defaults:
            raise ValueError(
                "Unsupported default node parameters: "
                f"{sorted(unknown_defaults)}"
            )

        defaults: dict[str, DistributionSpec] = {}

        for parameter_name, spec_payload in defaults_payload.items():
            if not isinstance(spec_payload, dict):
                raise ValueError(
                    f"Default parameter {parameter_name!r} "
                    "must be an object"
                )

            defaults[parameter_name] = (
                DistributionSpec.from_mapping(spec_payload)
            )

        rules_payload = payload.get("rules", [])
        if not isinstance(rules_payload, list):
            raise ValueError(
                "node_parameters.rules must be a list"
            )

        rules = tuple(
            NodeGroundTruthRule.from_mapping(
                rule_payload,
                index=index,
            )
            for index, rule_payload in enumerate(rules_payload)
            if isinstance(rule_payload, dict)
        )

        if len(rules) != len(rules_payload):
            raise ValueError(
                "Every item in node_parameters.rules must be an object"
            )

        return cls(
            defaults=defaults,
            rules=rules,
        )


@dataclass(frozen=True, slots=True)
class GroundTruthProfile:
    """Versioned description of synthetic mechanistic ground truth."""

    schema_version: int
    name: str
    description: str
    node_parameters: NodeParameterProfile

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "GroundTruthProfile":
        allowed = {
            "schema_version",
            "name",
            "description",
            "node_parameters",
        }

        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"Unknown profile fields: {sorted(unknown)}"
            )

        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError(
                "Only synthetic ground-truth profile "
                "schema_version 1 is currently supported"
            )

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Profile requires a non-empty string 'name'"
            )

        description = payload.get("description", "")
        if not isinstance(description, str):
            raise ValueError(
                "Profile description must be a string"
            )

        node_payload = payload.get("node_parameters")
        if not isinstance(node_payload, dict):
            raise ValueError(
                "Profile requires a 'node_parameters' object"
            )

        return cls(
            schema_version=1,
            name=name,
            description=description,
            node_parameters=NodeParameterProfile.from_mapping(
                node_payload
            ),
        )


def load_ground_truth_profile(
    path: str | Path,
) -> GroundTruthProfile:
    """Load and validate a versioned JSON ground-truth profile."""

    profile_path = Path(path).expanduser().resolve()

    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth profile does not exist: {profile_path}"
        )

    try:
        payload: Any = json.loads(
            profile_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in ground-truth profile "
            f"{profile_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Ground-truth profile root must be a JSON object"
        )

    return GroundTruthProfile.from_mapping(payload)


def _selector_array(
    anatomy: SyntheticAnatomy,
    selector: str,
) -> np.ndarray:
    if selector == "region":
        return np.asarray(
            anatomy.region_names,
            dtype=object,
        )

    if selector == "hemisphere":
        return np.asarray(
            anatomy.hemisphere,
            dtype=object,
        )

    if selector == "layer":
        return np.asarray(
            [
                anatomy.layer_names[int(index)]
                for index in anatomy.layer_index
            ],
            dtype=object,
        )

    if selector == "column_id":
        return np.asarray(
            anatomy.column_ids,
            dtype=np.int64,
        )

    if selector == "network_id":
        return np.asarray(
            anatomy.network_ids,
            dtype=np.int64,
        )

    if selector == "label":
        return np.asarray(
            anatomy.labels,
            dtype=object,
        )

    raise KeyError(
        f"Unsupported node selector {selector!r}"
    )


def _rule_mask(
    anatomy: SyntheticAnatomy,
    rule: NodeGroundTruthRule,
) -> np.ndarray:
    mask = np.ones(
        anatomy.num_nodes,
        dtype=bool,
    )

    for selector, expected in rule.match.items():
        actual = _selector_array(
            anatomy,
            selector,
        )

        expected_values = (
            expected
            if isinstance(expected, list)
            else [expected]
        )

        mask &= np.isin(
            actual,
            expected_values,
        )

    return mask


def apply_node_ground_truth_profile(
    profile: GroundTruthProfile,
    anatomy: SyntheticAnatomy,
    base_parameters: Mapping[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Overlay profile-defined parameters on procedural ground truth.

    Parameters omitted from the profile retain their procedural values.
    Rules are applied in listed order, so later rules can deliberately
    override earlier, broader rules.
    """

    n_nodes = anatomy.num_nodes

    result: dict[str, np.ndarray] = {}

    for parameter_name, values in base_parameters.items():
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.shape != (n_nodes,):
            raise ValueError(
                f"Base parameter {parameter_name!r} must have "
                f"shape {(n_nodes,)}, got {array.shape}"
            )

        result[parameter_name] = array.copy()

    node_profile = profile.node_parameters

    for parameter_name, spec in node_profile.defaults.items():
        if parameter_name not in result:
            raise ValueError(
                f"Profile refers to unavailable node parameter "
                f"{parameter_name!r}"
            )

        result[parameter_name] = spec.sample(
            rng,
            n_nodes,
        )

    for rule in node_profile.rules:
        mask = _rule_mask(
            anatomy,
            rule,
        )

        count = int(mask.sum())

        if count == 0:
            if rule.required:
                raise ValueError(
                    f"Ground-truth rule {rule.name!r} "
                    "matched no synthetic nodes"
                )
            continue

        for parameter_name, spec in rule.parameters.items():
            if parameter_name not in result:
                raise ValueError(
                    f"Rule {rule.name!r} refers to unavailable "
                    f"parameter {parameter_name!r}"
                )

            result[parameter_name][mask] = spec.sample(
                rng,
                count,
            )

    return result
