"""Synthetic cortical geometry, laminar labels, and hierarchy construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SyntheticFMRIConfig


@dataclass(slots=True)
class SyntheticAnatomy:
    """Node-wise synthetic anatomy used by the mechanistic generator."""

    coordinates_mm: np.ndarray
    hierarchy: np.ndarray
    spatial_gradient: np.ndarray
    layer_index: np.ndarray
    layer_names: tuple[str, ...]
    column_ids: np.ndarray
    network_ids: np.ndarray
    hemisphere: np.ndarray
    region_names: tuple[str, ...]
    labels: tuple[str, ...]
    sensory_nodes: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.coordinates_mm.shape[0])


def _region_for_hierarchy(value: float) -> tuple[str, int]:
    if value < 0.25:
        return "S1_Postcentral", 0
    if value < 0.50:
        return "M1_Precentral", 1
    if value < 0.75:
        return "S2_ParietalOperculum", 2
    return "Association_Parietal", 3


def build_synthetic_anatomy(config: SyntheticFMRIConfig) -> SyntheticAnatomy:
    """Create bilateral, layered pseudo-cortical nodes with a known hierarchy.

    Coordinates resemble MNI ranges only so the existing GBB coordinate-based
    stimulus selector can be exercised. They are not derived from a participant,
    atlas, or anatomical scan.
    """
    config.validate()
    rng = np.random.default_rng(config.seed)
    layer_names = tuple(config.cortical_layers)
    #nodes_per_column = len(layer_names)

    coords: list[list[float]] = []
    hierarchy: list[float] = []
    spatial_gradient: list[float] = []
    layer_index: list[int] = []
    column_ids: list[int] = []
    network_ids: list[int] = []
    hemispheres: list[str] = []
    region_names: list[str] = []
    labels: list[str] = []

    # Reserve the first left-hemisphere sensory column near the current GBB
    # default stimulus coordinate (-42, -25, 55).
    per_hemi = int(np.ceil(config.num_columns / 2))
    for column in range(config.num_columns):
        hemisphere = "Left" if column % 2 == 0 else "Right"
        hemi_sign = -1.0 if hemisphere == "Left" else 1.0
        local_index = column // 2
        fraction = local_index / max(1, per_hemi - 1)

        # Posterior-to-anterior and inferior-to-superior gradients provide a
        # known spatial axis that partly aligns with the hierarchy.
        y = -25.0 + 44.0 * fraction
        z = 55.0 - 18.0 * fraction + 3.0 * np.sin(np.pi * fraction)
        x = hemi_sign * (42.0 + 10.0 * fraction)
        x += float(rng.normal(0.0, 1.0))
        y += float(rng.normal(0.0, 1.0))
        z += float(rng.normal(0.0, 0.8))

        base_hierarchy = np.clip(fraction + rng.normal(0.0, 0.025), 0.0, 1.0)
        region_name, network_id = _region_for_hierarchy(base_hierarchy)

        for layer_name in layer_names:
            li = config.layer_order[layer_name]
            # Layer offsets are small enough to represent a cortical column but
            # large enough to create distinct labelled parcels in the toy atlas.
            radial_offset = (li - np.mean(list(config.layer_order.values()))) * 2.4
            layer_coord = [x, y, z + radial_offset]
            #node_index = len(coords)
            coords.append(layer_coord)
            hierarchy.append(float(np.clip(base_hierarchy + 0.025 * (li - 1), 0.0, 1.0)))
            spatial_gradient.append(float((y + 25.0) / 44.0))
            layer_index.append(li)
            column_ids.append(column + 1)
            network_ids.append(network_id)
            hemispheres.append(hemisphere)
            region_names.append(region_name)
            labels.append(
                f"{hemisphere}_{region_name}_Column_{column + 1:02d}_{layer_name.capitalize()}"
            )

    coords_array = np.asarray(coords, dtype=np.float64)
    hierarchy_array = np.asarray(hierarchy, dtype=np.float64)
    gradient_array = np.clip(np.asarray(spatial_gradient, dtype=np.float64), 0.0, 1.0)
    layer_array = np.asarray(layer_index, dtype=np.int64)
    network_array = np.asarray(network_ids, dtype=np.int64)
    column_array = np.asarray(column_ids, dtype=np.int64)
    hemisphere_array = np.asarray(hemispheres, dtype="U8")

    sensory_mask = np.array(
        [
            ("S1_Postcentral" in region_names[index])
            and hemispheres[index] == "Left"
            and layer_names[layer_array[index]] == "middle"
            for index in range(len(labels))
        ],
        dtype=bool,
    )
    if not sensory_mask.any():
        # Deterministic fallback for very small custom geometries.
        sensory_mask[np.argmin(np.linalg.norm(coords_array - np.array([-42.0, -25.0, 55.0]), axis=1))] = True

    return SyntheticAnatomy(
        coordinates_mm=coords_array,
        hierarchy=hierarchy_array,
        spatial_gradient=gradient_array,
        layer_index=layer_array,
        layer_names=layer_names,
        column_ids=column_array,
        network_ids=network_array,
        hemisphere=hemisphere_array,
        region_names=tuple(region_names),
        labels=tuple(labels),
        sensory_nodes=sensory_mask,
    )
