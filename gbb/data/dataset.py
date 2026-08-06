"""PyTorch dataset for windowed labelled-mask fMRI time series."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from nilearn.maskers import NiftiLabelsMasker
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from gbb.config import cfg, config
from gbb.data.fmri import load_fmri_run
from gbb.data.masks import (
    compute_centroids,
    get_region_ids,
    get_region_labels,
    load_column_ids,
    load_mask,
    load_mask_metadata,
)
from gbb.data.stimulus import StimulusLoader
from gbb.data.targeting import make_sensory_mask, select_sensory_indices


class NiftiLaminarDataset(Dataset):
    """Load multiple fMRI runs and expose run-boundary-safe windows.

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

        self.window_size = int(config.WINDOW_SIZE if window_size is None else window_size)
        self.prediction_horizon = int(config.PREDICTION_HORIZON)
        self.tr = float(config.TR)
        self.is_training = run_type.lower() == "train"

        if self.window_size <= 0 or self.prediction_horizon <= 0:
            raise ValueError("window_size and prediction_horizon must be positive")
        if self.tr <= 0:
            raise ValueError("TR must be positive")

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

        self.stimulus_loader = StimulusLoader(
            tr=self.tr,
            settings=cfg.stimulus,
            use_hemodynamic_head=bool(cfg.hemodynamics.use_head),
        )

        if self.stimulus_loader.stimulus_mode == "NONE":
            self.sensory_indices: list[int] = []
        else:
            configured_regions = (
                list(config.SENSORY_REGIONS) if sensory_regions is None else list(sensory_regions)
            )
            self.sensory_indices = select_sensory_indices(
                region_labels=self.region_labels,
                coordinates=self.coords,
                sensory_regions=configured_regions,
                excluded_regions=list(config.EXCLUDED_REGIONS),
                injection_mode=str(config.STIMULUS_INJECTION_MODE),
                target_mni=tuple(config.STIMULUS_MNI_COORDS),
                radius_mm=float(config.STIMULUS_RADIUS_MM),
                selection_policy=str(config.SENSORY_SELECTION_POLICY),
                allow_all_nodes=bool(config.ALLOW_ALL_NODES_STIMULUS),
            )

        self.sensory_mask = make_sensory_mask(
            num_nodes=self.num_nodes,
            sensory_indices=self.sensory_indices,
        )

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
        self.distance_matrix = torch.as_tensor(
            distance_matrix,
            dtype=torch.float32,
        )

        sigma = float(
            getattr(
                config,
                "ADJACENCY_SIGMA",
                config.SMOOTHNESS_SIGMA_MM,
            )
        )
        if sigma <= 0:
            raise ValueError("ADJACENCY_SIGMA must be positive")

        adjacency = np.exp(-(distance_matrix**2) / (2.0 * sigma**2))
        adjacency[adjacency < 0.01] = 0.0
        np.fill_diagonal(adjacency, 1.0)
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)

        self.node_region_ids = self._make_region_group_ids(self.region_labels)
        self.column_ids = load_column_ids(
            columnar_mask_path=Path(str(config.COLUMNAR_MASK_FILE)),
            parcellation_img=self.parcellation_img,
            mask_data=mask_data,
            region_ids=self.region_ids,
            node_region_ids=self.node_region_ids,
            policy=str(getattr(config, "COLUMNAR_MASK_POLICY", "ERROR")),
        )

    def _load_runs(
        self,
        runs: list[dict[str, str]],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        series_parts: list[np.ndarray] = []
        stimulus_parts: list[np.ndarray] = []
        run_lengths: list[int] = []

        iterator = tqdm(runs, desc="Processing runs", leave=False) if len(runs) > 1 else runs
        for item in iterator:
            if "fmri" not in item:
                raise KeyError("Each run entry must contain an 'fmri' path")

            fmri_data = load_fmri_run(
                fmri_path=Path(item["fmri"]),
                masker=self.masker,
                parcellation_img=self.parcellation_img,
                expected_nodes=self.num_nodes,
            )
            run_length = int(fmri_data.shape[0])
            stimulus = self.stimulus_loader.load(
                item,
                n_scans=run_length,
            )

            series_parts.append(fmri_data)
            stimulus_parts.append(stimulus)
            run_lengths.append(run_length)

        return (
            np.concatenate(series_parts, axis=0).astype(np.float32),
            np.concatenate(stimulus_parts, axis=0).astype(np.float32),
            run_lengths,
        )

    def _build_valid_start_indices(self) -> list[int]:
        starts: list[int] = []
        required_length = self.window_size + self.prediction_horizon

        for offset, run_length in zip(
            self.run_offsets,
            self.run_lengths,
            strict=True,
        ):
            maximum_local_start = run_length - required_length
            if maximum_local_start >= 0:
                starts.extend(range(offset, offset + maximum_local_start + 1))

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

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.valid_start_indices[index]
        history_stop = start + self.window_size
        future_stop = history_stop + self.prediction_horizon

        fmri_history = torch.from_numpy(self.time_series[start:history_stop])
        stimulus_history = torch.from_numpy(self.stim_drive[start:history_stop])
        fmri_future = torch.from_numpy(self.time_series[history_stop:future_stop])

        return (
            fmri_history.float(),
            stimulus_history.float(),
            fmri_future.float(),
        )
