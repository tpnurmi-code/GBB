"""Aggregation of node-level outputs to coarser anatomical labels."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


def condense_to_anatomy(data_matrix, fine_labels, coarse_coords_file=None):
    """Average node-level data within repeated anatomical labels.

    Parameters
    ----------
    data_matrix:
        A node vector ``(N,)``, square connection matrix ``(N, N)``, or a
        node-by-time/time-by-node matrix.
    fine_labels:
        One anatomical label per node.
    coarse_coords_file:
        Retained only for backward API compatibility. Coordinate calculation is
        intentionally handled by the plotting layer because labels alone cannot
        establish a reliable atlas-ID-to-name mapping.
    """
    del coarse_coords_file
    values = np.asarray(data_matrix)
    labels = [str(label) for label in fine_labels]
    if not labels:
        raise ValueError("fine_labels is empty")
    node_count = len(labels)

    # Preserve first appearance rather than alphabetically reordering anatomy.
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    unique_labels = list(groups)

    if values.ndim == 1:
        if values.shape[0] != node_count:
            raise ValueError(
                f"Vector length {values.shape[0]} does not match {node_count} labels"
            )
        condensed = np.asarray(
            [np.nanmean(values[groups[label]]) for label in unique_labels],
            dtype=float,
        )
        return condensed, unique_labels

    if values.ndim != 2:
        raise ValueError(f"Expected a one- or two-dimensional array, got {values.shape}")

    if values.shape == (node_count, node_count):
        condensed = np.zeros((len(unique_labels), len(unique_labels)), dtype=float)
        for row_index, row_label in enumerate(unique_labels):
            for column_index, column_label in enumerate(unique_labels):
                block = values[np.ix_(groups[row_label], groups[column_label])]
                condensed[row_index, column_index] = np.nanmean(block)
        return condensed, unique_labels

    if values.shape[1] == node_count:
        time_by_node = values
    elif values.shape[0] == node_count:
        time_by_node = values.T
    else:
        raise ValueError(
            f"Neither axis of {values.shape} matches the {node_count} labels"
        )
    condensed_time = np.column_stack(
        [np.nanmean(time_by_node[:, groups[label]], axis=1) for label in unique_labels]
    )
    # Preserve the historical plotting convention: regions x time.
    return condensed_time.T, unique_labels