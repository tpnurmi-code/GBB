"""PyTorch dataset for windowed labelled-mask fMRI time series."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import torch
from nilearn.glm.first_level import compute_regressor
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from gbb.config import config
from gbb.data.hrf import canonical_cbv_response, canonical_hrf_response
from gbb.data.masks import (
    compute_centroids,
    get_region_ids,
    get_region_labels,
    load_mask,
    load_mask_metadata,
)


class NiftiLaminarDataset(Dataset):
    """Load multiple fMRI runs and expose non-overlapping-boundary windows.

    Each item is ``(fmri_history, stimulus_history, fmri_future)`` with shapes
    ``(window_size, nodes)``, ``(window_size, channels)``, and
    ``(prediction_horizon, nodes)``.
    """

    def __init__(
        self,
        data_list: Iterable[dict[str, str]],
        mask_img,
        window_size: int | None = None,
        run_type: str = "train",
        sensory_regions: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.window_size = int(
            config.WINDOW_SIZE if window_size is None else window_size
        )
        self.prediction_horizon = int(config.PREDICTION_HORIZON)
        self.tr = float(config.TR)
        self.is_training = run_type.lower() == "train"

        self.stimulus_mode = self._read_policy(
            "STIMULUS_MODE",
            {"EVENTS", "DENSE", "NONE"},
            default="NONE",
        )
        self.missing_stimulus_policy = self._read_policy(
            "MISSING_STIMULUS_POLICY",
            {"ERROR", "WARN", "ZEROS"},
            default="ERROR",
        )
        self.stimulus_shape_policy = self._read_policy(
            "STIMULUS_SHAPE_POLICY",
            {"ERROR", "WARN", "COERCE"},
            default="ERROR",
        )
        self.sensory_selection_policy = self._read_policy(
            "SENSORY_SELECTION_POLICY",
            {"STRICT", "WARN", "FALLBACK"},
            default="STRICT",
        )

        self.stimulus_channels = int(config.STIMULUS_INPUT_CHANNELS)
        self.require_nonzero_stimulus = bool(
            getattr(config, "REQUIRE_NONZERO_STIMULUS", True)
        )
        self.allow_all_nodes_stimulus = bool(
            getattr(config, "ALLOW_ALL_NODES_STIMULUS", False)
        )
        self.required_trial_type = getattr(
            config,
            "REQUIRED_TRIAL_TYPE",
            "hand_movement",
        )
        self.dense_stimulus_key = getattr(
            config,
            "DENSE_STIMULUS_KEY",
            "stimulus",
        )

        if self.stimulus_channels <= 0:
            raise ValueError("STIMULUS_INPUT_CHANNELS must be positive")

        if self.window_size <= 0 or self.prediction_horizon <= 0:            raise ValueError("window_size and prediction_horizon must be positive")

        self.parcellation_img, mask_data, affine, mask_path = load_mask(mask_img)
        self.mask_path = mask_path
        self.masker = NiftiLabelsMasker(
            labels_img=self.parcellation_img,
            standardize=False,
            detrend=config.DETREND,
            t_r=self.tr,
            low_pass=config.LOW_PASS,
            high_pass=config.HIGH_PASS,
        )
        self.masker.fit()

        self.region_ids = get_region_ids(mask_data)
        self.num_nodes = int(len(self.region_ids))
        if self.num_nodes == 0:
            raise ValueError("The labelled mask contains no non-zero regions")

	
        metadata = load_mask_metadata(mask_path)
        self.coords = compute_centroids(
            mask_data=mask_data,
            affine=affine,
            region_ids=self.region_ids,
            metadata=metadata,
        )
        self.node_coords = self.coords.copy()
        self.region_labels = get_region_labels(self.region_ids, metadata)

        if self.stimulus_mode == "NONE":
            self.sensory_indices = []
        else:
            configured_regions = (
                list(config.SENSORY_REGIONS)
                if sensory_regions is None
                else list(sensory_regions)
            )
            self.sensory_indices = self._select_sensory_indices(
                configured_regions
            )
        
        sensory_mask_np = np.zeros((self.num_nodes, 1), dtype=np.float32)
        sensory_mask_np[self.sensory_indices, 0] = 1.0
        self.sensory_mask = torch.from_numpy(sensory_mask_np)

        runs = list(data_list)
        if not runs:
            raise ValueError("data_list is empty; no fMRI runs were supplied")

        self.time_series, self.stim_drive, self.run_lengths = self._load_runs(runs)
        self.total_time = int(self.time_series.shape[0])
        self.run_offsets = np.cumsum([0] + self.run_lengths[:-1]).tolist()
        self.valid_start_indices = self._build_valid_start_indices()
        if not self.valid_start_indices:
            raise ValueError(
                "No run is long enough for window_size + prediction_horizon "
                f"({self.window_size} + {self.prediction_horizon})."
            )

        distance_matrix = cdist(self.coords, self.coords, metric="euclidean")
        self.distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
        sigma = float(getattr(config, "ADJACENCY_SIGMA", config.SMOOTHNESS_SIGMA_MM))
        adjacency = np.exp(-(distance_matrix**2) / (2.0 * sigma**2))
        adjacency[adjacency < 0.01] = 0.0
        np.fill_diagonal(adjacency, 1.0)
        self.adjacency = torch.tensor(adjacency, dtype=torch.float32)

        self.node_region_ids = self._make_region_group_ids(self.region_labels)
        self.column_ids = self._load_column_ids(mask_data)

  
    def _load_runs(
        self,
        runs: list[dict[str, str]],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        series_parts: list[np.ndarray] = []
        stimulus_parts: list[np.ndarray] = []
        run_lengths: list[int] = []

        iterator = tqdm(runs, desc="Processing runs", leave=False) if len(runs) > 1 else runs
        for item in iterator:
           fmri_data = load_fmri_run(
           fmri_path=Path(item["fmri"]),
           masker=self.masker,
           parcellation_img=self.parcellation_img,
           expected_nodes=self.num_nodes,
           )
           stimulus = self.stimulus_loader.load(
           item,
           n_scans=fmri_data.shape[0],
           )

            node_std = np.std(fmri_data, axis=0)
            constant_nodes = np.flatnonzero(node_std <= 1e-8)

            if constant_nodes.size:
                raise ValueError(
                    f"Extracted fMRI data from {fmri_path} contains "
                    f"{constant_nodes.size} constant or near-constant nodes. "
                    "First affected node indices: "
                    f"{constant_nodes[:10].tolist()}"
                )

            fmri_data = zscore(
                fmri_data,
                axis=0,
            ).astype(np.float32)

            if not np.all(np.isfinite(fmri_data)):
                raise RuntimeError(
                    f"Z-scoring produced non-finite values for {fmri_path}"
                )

            run_length = int(fmri_data.shape[0])
            stimulus = self._load_stimulus(item, run_length)

            series_parts.append(fmri_data)
            stimulus_parts.append(stimulus)
            run_lengths.append(run_length)

        return (
            np.concatenate(series_parts, axis=0).astype(np.float32),
            np.concatenate(stimulus_parts, axis=0).astype(np.float32),
            run_lengths,
        )
    

    def _should_convolve_stimulus(self) -> bool:
        should_convolve = bool(config.CONVOLVE_STIMULUS)

        if bool(config.USE_HEMODYNAMIC_HEAD):
            should_convolve = (
                should_convolve
                and bool(config.ALLOW_STIMULUS_PRECONV_WITH_HEMO)
            )

        return should_convolve

    def _stimulus_response_kernel(self) -> np.ndarray:
        response_function = str(config.RESPONSE_FUNCTION).lower()

        if response_function == "cbv":
            return canonical_cbv_response(self.tr, 30.0)

        if response_function == "hrf":
            return canonical_hrf_response(30.0, tr=self.tr)

        if response_function == "uniform":
            # Identity kernel: preserve the sampled stimulus unchanged.
            return np.ones(1, dtype=np.float32)

        raise ValueError(
            f"Unknown RESPONSE_FUNCTION: {config.RESPONSE_FUNCTION!r}"
        )


    def _build_valid_start_indices(self) -> list[int]:
        starts: list[int] = []
        for offset, run_length in zip(self.run_offsets, self.run_lengths):
            maximum_local_start = run_length - (
                self.window_size + self.prediction_horizon
            )
            if maximum_local_start >= 0:
                starts.extend(
                    range(offset, offset + maximum_local_start + 1)
                )
        return starts

    @staticmethod
    def _make_region_group_ids(labels: list[str]) -> torch.Tensor:
        unique_labels = {label: index for index, label in enumerate(sorted(set(labels)))}
        return torch.tensor(
            [unique_labels[label] for label in labels],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def __getitem__(self, index: int):
        start = self.valid_start_indices[index]
        history_stop = start + self.window_size
        future_stop = history_stop + self.prediction_horizon

        fmri_history = torch.from_numpy(self.time_series[start:history_stop])
        stimulus_history = torch.from_numpy(self.stim_drive[start:history_stop])
        fmri_future = torch.from_numpy(self.time_series[history_stop:future_stop])
        return fmri_history.float(), stimulus_history.float(), fmri_future.float()