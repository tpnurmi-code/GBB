"""Connectivity and matrix visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting


def _mask_centroids(mask_path) -> np.ndarray:
    image = nib.load(str(mask_path))
    data = image.get_fdata()
    labels = np.sort(np.unique(data[data != 0]))
    return np.asarray(
        [
            nib.affines.apply_affine(
                image.affine,
                np.argwhere(data == label).mean(axis=0),
            )
            for label in labels
        ],
        dtype=float,
    )


def visualize_network(
    model,
    mask_img_path,
    save_path=None,
    adjacency_matrix=None,
) -> None:
    """Plot a sparse effective-interaction graph on a glass brain."""
    raw_model = model.module if hasattr(model, "module") else model
    node_count = int(raw_model.num_nodes)
    coordinates = _mask_centroids(mask_img_path)
    if len(coordinates) != node_count:
        raise ValueError(
            f"Mask contains {len(coordinates)} labels; model contains {node_count} nodes"
        )

    if adjacency_matrix is None:
        candidate = None
        if save_path is not None:
            candidate = Path(save_path).parent / "EffectiveInteraction_target_by_source.npy"
        if candidate is not None and candidate.exists():
            adjacency_matrix = np.load(candidate)
        else:
            adjacency_matrix = np.zeros((node_count, node_count), dtype=float)
    adjacency = np.asarray(adjacency_matrix, dtype=float)
    if adjacency.shape != (node_count, node_count):
        raise ValueError(f"Unexpected adjacency shape {adjacency.shape}")

    nonzero = np.abs(adjacency[np.isfinite(adjacency)])
    threshold = "99%" if np.any(nonzero > 0) else None
    display = plotting.plot_connectome(
        adjacency_matrix=adjacency,
        node_coords=coordinates,
        node_size=15,
        edge_threshold=threshold,
        display_mode="lzr",
        colorbar=np.any(nonzero > 0),
        title="Learned effective interaction network",
        output_file=str(save_path) if save_path is not None else None,
    )
    if save_path is None:
        display.show()
    display.close()


def visualize_chord_diagram(
    adj_matrix,
    node_coords,
    save_path,
    labels=None,
) -> None:
    """Save a circular connectivity diagram.

    The function no longer invents anatomical groups from an unrelated coarse
    mask. Pass an already-condensed matrix and its corresponding labels when a
    coarse anatomical view is wanted. ``node_coords`` is retained for backward
    compatibility and is otherwise unused.
    """
    del node_coords
    matrix = np.asarray(adj_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adj_matrix must be square")
    names = list(labels) if labels is not None else [f"Node_{i}" for i in range(matrix.shape[0])]
    if len(names) != matrix.shape[0]:
        raise ValueError("Number of labels must match matrix size")

    try:
        from mne.viz import plot_connectivity_circle
    except ImportError as exc:
        raise ImportError(
            "MNE is required for chord diagrams: pip install mne"
        ) from exc

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    plot_connectivity_circle(
        matrix,
        names,
        ax=axis,
        show=False,
        n_lines=min(100, matrix.size),
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def export_to_brainnet(
    adj_matrix,
    coords,
    labels,
    tau_values,
    save_dir,
    top_k: int = 30,
) -> tuple[Path, Path]:
    """Export BrainNet Viewer ``.node`` and ``.edge`` files."""
    matrix = np.asarray(adj_matrix, dtype=float)
    coordinates = np.asarray(coords, dtype=float)
    tau = np.asarray(tau_values, dtype=float).reshape(-1)
    names = [str(label).replace(" ", "_") for label in labels]
    node_count = matrix.shape[0]
    if matrix.shape != (node_count, node_count):
        raise ValueError("adj_matrix must be square")
    if coordinates.shape != (node_count, 3) or tau.size != node_count or len(names) != node_count:
        raise ValueError("coords, labels, and tau_values must all match adj_matrix")

    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    absolute = np.abs(matrix)
    positive = absolute[absolute > 0]
    if positive.size > top_k:
        threshold = np.partition(positive, -top_k)[-top_k]
    elif positive.size:
        threshold = positive.min()
    else:
        threshold = np.inf
    sparse = np.where(absolute >= threshold, matrix, 0.0)

    node_path = output / "network.node"
    with node_path.open("w", encoding="utf-8") as handle:
        node_sizes = absolute.sum(axis=0)
        for coordinate, color_value, size_value, label in zip(
            coordinates, tau, node_sizes, names, strict=True
        ):
            handle.write(
                f"{coordinate[0]:.4f} {coordinate[1]:.4f} {coordinate[2]:.4f} "
                f"{color_value:.4f} {size_value:.4f} {label}\n"
            )
    edge_path = output / "network.edge"
    np.savetxt(edge_path, sparse, fmt="%.6f", delimiter="\t")
    return node_path, edge_path


def visualize_tau_sorted_matrix(
    adj_matrix,
    tau_values,
    labels,
    save_dir,
) -> Path:
    """Plot the effective-interaction matrix after ordering nodes by tau."""
    matrix = np.asarray(adj_matrix, dtype=float)
    tau = np.asarray(tau_values, dtype=float).reshape(-1)
    names = [str(label) for label in labels]
    if matrix.shape != (tau.size, tau.size) or len(names) != tau.size:
        raise ValueError("Matrix, tau vector, and labels have incompatible sizes")
    order = np.argsort(tau)
    sorted_matrix = matrix[np.ix_(order, order)]

    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "effective_interaction_tau_sorted.png"
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(sorted_matrix, aspect="auto", origin="lower")
    figure.colorbar(image, ax=axis, label="Signed effective interaction")
    axis.set_title("Effective interaction ordered by CfC time constant")
    axis.set_xlabel("Source nodes: fast to slow tau")
    axis.set_ylabel("Target nodes: fast to slow tau")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path