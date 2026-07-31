"""Prediction and in-silico stimulus-response visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def visualize_prediction_dynamics(model, dataset, device, save_path) -> Path:
    """Plot a stimulus-targeted node and the best/worst predicted nodes."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    selected_batch = None
    fallback_batch = None
    for batch in loader:
        fallback_batch = batch
        if batch[1].abs().sum().item() > 0:
            selected_batch = batch
            break
    batch = selected_batch or fallback_batch
    if batch is None:
        raise ValueError("Dataset is empty")

    history, stimulus, target = (tensor.to(device) for tensor in batch[:3])
    adjacency = dataset.adjacency.to(device)
    raw_model = _unwrap_model(model)
    raw_model.eval()
    with torch.no_grad():
        prediction, _, _, _ = raw_model(history, stimulus, adjacency)

    history_np = history[0].cpu().numpy()
    truth_np = target[0].cpu().numpy()
    prediction_np = prediction[0].cpu().numpy()
    node_count = history_np.shape[1]
    correlations = np.zeros(node_count, dtype=float)
    for node in range(node_count):
        if np.std(prediction_np[:, node]) < 1e-8 or np.std(truth_np[:, node]) < 1e-8:
            correlations[node] = 0.0
        else:
            correlations[node] = np.corrcoef(
                truth_np[:, node], prediction_np[:, node]
            )[0, 1]
    correlations = np.nan_to_num(correlations)

    sensory = np.flatnonzero(np.asarray(dataset.sensory_mask).reshape(-1) > 0)
    stimulus_node = int(sensory[0]) if sensory.size else 0
    selected_nodes = [stimulus_node, int(np.argmax(correlations)), int(np.argmin(correlations))]
    titles = ["Stimulus-targeted node", "Best future fit", "Worst future fit"]
    history_time = np.arange(history_np.shape[0])
    future_time = np.arange(history_np.shape[0], history_np.shape[0] + truth_np.shape[0])

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, node, title in zip(axes, selected_nodes, titles, strict=True):
        axis.plot(history_time, history_np[:, node], label="Observed history")
        axis.plot(future_time, truth_np[:, node], label="Observed future")
        axis.plot(future_time, prediction_np[:, node], linestyle="--", label="GBB prediction")
        axis.axvline(history_time[-1], linestyle=":")
        axis.set_title(f"{title}: node {node}, future r={correlations[node]:.2f}")
        axis.set_ylabel("Standardized signal")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("TR index")
    figure.tight_layout()

    path = Path(save_path)
    if path.suffix == "":
        path = path / "prediction_dynamics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def simulate_spreading_activation(
    model,
    dataset,
    duration: int | None = None,
    pulse_time: int = 5,
):
    """Compare zero-input predictions with and without a synthetic stimulus.

    The current model predicts only the configured future horizon, so the output
    is a node-by-future-step response map rather than a full causal simulation.
    """
    raw_model = _unwrap_model(model)
    device = next(raw_model.parameters()).device
    duration = int(duration or raw_model.time_points)
    if duration != raw_model.time_points:
        raise ValueError(
            f"Model was built for {raw_model.time_points} input steps, not {duration}"
        )
    node_count = int(dataset.num_nodes)
    baseline_input = torch.zeros(1, duration, node_count, device=device)
    baseline_stimulus = torch.zeros(
        1,
        duration,
        raw_model.stim_encoder.in_channels,
        device=device,
    )
    pulse_stimulus = baseline_stimulus.clone()
    start = min(max(0, int(pulse_time)), duration - 1)
    pulse_stimulus[:, start : min(duration, start + 2), :] = 5.0
    adjacency = dataset.adjacency.to(device)

    raw_model.eval()
    with torch.no_grad():
        baseline, _, _, _ = raw_model(baseline_input, baseline_stimulus, adjacency)
        stimulated, _, _, _ = raw_model(baseline_input, pulse_stimulus, adjacency)
    difference = (stimulated - baseline)[0].detach().cpu().numpy().T  # nodes,horizon
    time_to_peak = np.argmax(np.abs(difference), axis=1)
    order = np.argsort(time_to_peak)
    labels = [dataset.region_labels[index] for index in order]
    return difference[order], labels, time_to_peak


def plot_spreading_activation(activation_map, labels, save_path) -> Path:
    matrix = np.asarray(activation_map, dtype=float)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, max(6, min(18, len(labels) * 0.12))))
    image = axis.imshow(matrix, aspect="auto", origin="lower")
    figure.colorbar(image, ax=axis, label="Stimulated minus baseline prediction")
    axis.set_title("Synthetic-stimulus prediction difference")
    axis.set_xlabel("Future prediction step")
    axis.set_ylabel("Nodes ordered by time to peak")
    if len(labels) <= 40:
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path