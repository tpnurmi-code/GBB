"""High-level post-training export orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gbb.analysis.anatomical_aggregation import condense_to_anatomy
from gbb.analysis.map_export import save_mechanistic_atlas
from gbb.config import config
from gbb.visualization.connectome import (
    export_to_brainnet,
    visualize_chord_diagram,
    visualize_tau_sorted_matrix,
)
from gbb.visualization.timeseries import (
    plot_spreading_activation,
    simulate_spreading_activation,
)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def gather_visualization_data(model, dataset, mask_path, dataloader=None):
    """Return effective interaction, node coordinates, labels, and tau values."""
    del mask_path  # dataset coordinates are ID-aligned and therefore preferred.
    raw_model = _unwrap_model(model)
    raw_model.eval()
    device = next(raw_model.parameters()).device

    adjacency_matrix = np.eye(dataset.num_nodes, dtype=float)
    if dataloader is not None:
        batch = next(iter(dataloader), None)
        if batch is not None:
            fmri, stimulus = batch[0].to(device), batch[1].to(device)
            adjacency = dataset.adjacency.to(device)
            with torch.no_grad():
                _, attention, _, _ = raw_model(fmri, stimulus, adjacency)
            if attention is not None:
                adjacency_matrix = attention.mean(dim=0).detach().cpu().numpy()

    coordinates = np.asarray(dataset.coords, dtype=float)
    labels = [str(label) for label in dataset.region_labels]
    tau_values = raw_model.cfc.get_tau_values().detach().cpu().numpy()
    if adjacency_matrix.shape != (len(labels), len(labels)):
        raise ValueError(
            f"Interaction matrix {adjacency_matrix.shape} does not match {len(labels)} labels"
        )
    return adjacency_matrix, coordinates, labels, tau_values


def visualize_results(
    model,
    ref_ds,
    world_size,
    all_runs,
    node_vals,
    test_loader,
    nowstring,
):
    """Generate the non-interactive outputs used after training.

    ``world_size`` is retained for call-site compatibility. Only rank zero calls
    this function, so no distributed gather is performed here.
    """
    del world_size
    output_directory = Path(config.RESULTS_DIR)
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = all_runs[0].get("mask", config.MASK_FILE) if all_runs else config.MASK_FILE

    map_directory = save_mechanistic_atlas(
        model=model,
        mask_path=mask_path,
        node_vals=node_vals,
        output_dir=output_directory,
        masker=ref_ds.masker,
        dataloader=test_loader,
        datestring=nowstring,
    )
    interaction, coordinates, labels, tau = gather_visualization_data(
        model,
        ref_ds,
        mask_path,
        dataloader=test_loader,
    )
    np.save(map_directory / "EffectiveInteraction_for_visualization.npy", interaction)
    export_to_brainnet(
        interaction,
        coordinates,
        labels,
        tau,
        map_directory,
    )
    visualize_tau_sorted_matrix(
        interaction,
        tau,
        labels,
        map_directory,
    )

    # A coarse view is only valid when repeated labels genuinely define the
    # grouping. Generic Region_1...Region_N labels therefore remain uncondensed.
    if len(set(labels)) < len(labels):
        coarse_interaction, coarse_labels = condense_to_anatomy(interaction, labels)
        visualize_chord_diagram(
            coarse_interaction,
            node_coords=None,
            save_path=map_directory / "effective_interaction_chord.png",
            labels=coarse_labels,
        )

    response_map, response_labels, _ = simulate_spreading_activation(
        model,
        ref_ds,
    )
    plot_spreading_activation(
        response_map,
        response_labels,
        map_directory / "synthetic_stimulus_prediction_difference.png",
    )
    return map_directory