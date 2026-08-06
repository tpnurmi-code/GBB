"""Anatomical sensory-node selection helpers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch


def select_sensory_indices(
    *,
    region_labels: Sequence[str],
    coordinates: np.ndarray,
    sensory_regions: Sequence[str],
    excluded_regions: Sequence[str],
    injection_mode: str,
    target_mni: Sequence[float],
    radius_mm: float,
    selection_policy: str,
    allow_all_nodes: bool,
) -> list[int]:
    """Select stimulus-target nodes under an explicit fallback policy."""
    labels = [str(label) for label in region_labels]
    num_nodes = len(labels)
    if num_nodes == 0:
        raise ValueError("region_labels is empty")

    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (num_nodes, 3):
        raise ValueError(f"coordinates has shape {coords.shape}; expected ({num_nodes}, 3)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates contains NaN or infinite values")

    target = np.asarray(target_mni, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_mni must contain exactly three finite coordinates")
    if radius_mm < 0:
        raise ValueError("radius_mm must be non-negative")

    mode = str(injection_mode).upper()
    valid_modes = {
        "REGION_NAME",
        "COORDINATES",
        "COORDS_REGION_INTERSECTION",
    }
    if mode not in valid_modes:
        raise ValueError(f"injection_mode must be one of {sorted(valid_modes)}; got {mode!r}")

    policy = str(selection_policy).upper()
    valid_policies = {"STRICT", "WARN", "FALLBACK"}
    if policy not in valid_policies:
        raise ValueError(
            f"selection_policy must be one of {sorted(valid_policies)}; got {policy!r}"
        )

    labels_lower = [label.lower() for label in labels]
    sensory_terms = [str(term).strip().lower() for term in sensory_regions if str(term).strip()]
    excluded_terms = [str(term).strip().lower() for term in excluded_regions if str(term).strip()]

    name_indices = [
        index
        for index, label in enumerate(labels_lower)
        if sensory_terms
        and any(term in label for term in sensory_terms)
        and not any(term in label for term in excluded_terms)
    ]

    distances = np.linalg.norm(coords - target, axis=1)
    sphere_indices = np.flatnonzero(distances <= radius_mm).tolist()

    if mode == "REGION_NAME":
        selected = name_indices
        fallback = sphere_indices
    elif mode == "COORDINATES":
        selected = sphere_indices
        fallback = name_indices
    else:
        name_set = set(name_indices)
        selected = [index for index in sphere_indices if index in name_set]
        posterior_sphere = [index for index in sphere_indices if coords[index, 1] < target[1]]
        fallback = posterior_sphere or sphere_indices or name_indices

    if not selected:
        details = (
            "No sensory nodes matched the configured targeting rule. "
            f"mode={mode}, "
            f"sensory_regions={list(sensory_regions)}, "
            f"name_matches={len(name_indices)}, "
            f"coordinate_matches={len(sphere_indices)}, "
            f"target={target.tolist()}, "
            f"radius_mm={radius_mm}"
        )

        if policy == "STRICT":
            raise ValueError(details)
        if policy == "WARN":
            warnings.warn(
                f"{details} Using the explicit fallback selector.",
                RuntimeWarning,
                stacklevel=2,
            )
        selected = fallback

    if not selected and allow_all_nodes:
        selected = list(range(num_nodes))

    if not selected:
        raise ValueError(
            "No sensory nodes were selected, including by the configured "
            "fallback. Set ALLOW_ALL_NODES_STIMULUS=True only for an "
            "intentional whole-network ablation."
        )

    selected = sorted(set(int(index) for index in selected))
    if any(index < 0 or index >= num_nodes for index in selected):
        raise ValueError("Sensory selection produced an out-of-range node index")

    if len(selected) == num_nodes and not allow_all_nodes:
        raise ValueError(
            "Sensory selection resolved to every model node. Set "
            "ALLOW_ALL_NODES_STIMULUS=True only for an intentional "
            "whole-network ablation."
        )

    return selected


def make_sensory_mask(
    *,
    num_nodes: int,
    sensory_indices: Sequence[int],
) -> torch.Tensor:
    """Return a binary ``(num_nodes, 1)`` float mask."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    indices = sorted(set(int(index) for index in sensory_indices))
    if any(index < 0 or index >= num_nodes for index in indices):
        raise ValueError(f"sensory_indices must be within [0, {num_nodes - 1}]")

    mask = torch.zeros((num_nodes, 1), dtype=torch.float32)
    if indices:
        mask[indices, 0] = 1.0
    return mask
