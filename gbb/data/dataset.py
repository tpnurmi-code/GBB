"""PyTorch dataset for windowed labelled-mask fMRI time series."""

from __future__ import annotations

import os
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
        self.window_size = int(config.WINDOW_SIZE if window_size is None else window_size)
        self.prediction_horizon = int(config.PREDICTION_HORIZON)
        self.tr = float(config.TR)
        self.is_training = run_type.lower() == "train"

        if self.window_size <= 0 or self.prediction_horizon <= 0:
            raise ValueError("window_size and prediction_horizon must be positive")

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

        self.sensory_indices = self._select_sensory_indices(
            sensory_regions=sensory_regions or list(config.SENSORY_REGIONS)
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

    def _select_sensory_indices(self, sensory_regions: list[str]) -> list[int]:
        mode = str(config.STIMULUS_INJECTION_MODE).upper()
        labels_lower = [label.lower() for label in self.region_labels]
        sensory_terms = [term.lower() for term in sensory_regions]
        excluded_terms = [term.lower() for term in config.EXCLUDED_REGIONS]

        name_indices = [
            index
            for index, label in enumerate(labels_lower)
            if any(term in label for term in sensory_terms)
            and not any(term in label for term in excluded_terms)
        ]

        target = np.asarray(config.STIMULUS_MNI_COORDS, dtype=float)
        distances = np.linalg.norm(self.coords - target, axis=1)
        sphere_indices = np.where(distances <= float(config.STIMULUS_RADIUS_MM))[0].tolist()

        if mode == "REGION_NAME":
            selected = name_indices
        elif mode == "COORDINATES":
            selected = sphere_indices
        elif mode == "COORDS_REGION_INTERSECTION":
            name_set = set(name_indices)
            selected = [index for index in sphere_indices if index in name_set]
            if not selected:
                # Metadata may contain only generic labels. Prefer geometrically
                # posterior sphere nodes before opening every input port.
                selected = [index for index in sphere_indices if self.coords[index, 1] < target[1]]
            if not selected:
                selected = sphere_indices
        else:
            raise ValueError(f"Unknown STIMULUS_INJECTION_MODE: {mode}")

        if not selected:
            selected = list(range(self.num_nodes))
        return sorted(set(int(index) for index in selected))

    def _load_runs(
        self,
        runs: list[dict[str, str]],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        series_parts: list[np.ndarray] = []
        stimulus_parts: list[np.ndarray] = []
        run_lengths: list[int] = []

        iterator = tqdm(runs, desc="Processing runs", leave=False) if len(runs) > 1 else runs
        for item in iterator:
            fmri_path = Path(item["fmri"])
            if not fmri_path.exists():
                raise FileNotFoundError(f"fMRI file not found: {fmri_path}")

            functional_img = nib.load(str(fmri_path))
            resampled = resample_to_img(
                source_img=functional_img,
                target_img=self.parcellation_img,
                interpolation="continuous",
                force_resample=True,
                copy_header=True,
            )
            fmri_data = np.asarray(self.masker.transform(resampled), dtype=np.float32)
            if fmri_data.ndim != 2 or fmri_data.shape[1] != self.num_nodes:
                raise ValueError(
                    f"Masker returned {fmri_data.shape}; expected (time, {self.num_nodes})"
                )

            fmri_data = zscore(fmri_data, axis=0, nan_policy="omit")
            fmri_data = np.nan_to_num(fmri_data, copy=False).astype(np.float32)
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

    def _load_stimulus(self, item: dict[str, str], run_length: int) -> np.ndarray:
        mode = str(config.STIMULUS_MODE).upper()
        if mode == "EVENTS":
            return self._load_events_stimulus(item.get("events", ""), run_length)
        if mode == "DENSE":
            dense_path = self._resolve_dense_stimulus_path(item)
            return self._load_dense_stimulus(dense_path, run_length)
        return np.zeros((run_length, int(config.STIMULUS_INPUT_CHANNELS)), dtype=np.float32)

    @staticmethod
    def _strip_nifti_suffix(path: Path) -> Path:
        text = str(path)
        if text.endswith(".nii.gz"):
            return Path(text[:-7])
        if text.endswith(".nii"):
            return Path(text[:-4])
        return path.with_suffix("")

    def _resolve_dense_stimulus_path(self, item: dict[str, str]) -> Path:
        extension = str(config.DENSE_STIMULUS_EXT)
        events_path = Path(item.get("events", ""))
        candidates: list[Path] = []
        if str(events_path):
            event_text = str(events_path)
            if event_text.endswith("_events.tsv"):
                candidates.append(Path(event_text[: -len("_events.tsv")] + extension))
        fmri_base = self._strip_nifti_suffix(Path(item["fmri"]))
        candidates.append(Path(str(fmri_base) + extension))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_events_stimulus(self, events_path: str | os.PathLike[str], n_scans: int) -> np.ndarray:
        path = Path(events_path)
        if not path.exists():
            return np.zeros((n_scans, int(config.STIMULUS_INPUT_CHANNELS)), dtype=np.float32)

        events = pd.read_csv(path, sep="\t")
        if events.empty or "onset" not in events:
            return np.zeros((n_scans, int(config.STIMULUS_INPUT_CHANNELS)), dtype=np.float32)

        if "trial_type" in events and (events["trial_type"] == "hand_movement").any():
            events = events[events["trial_type"] == "hand_movement"]

        onsets = events["onset"].to_numpy(dtype=float)
        durations = (
            events["duration"].to_numpy(dtype=float)
            if "duration" in events
            else np.zeros_like(onsets)
        )
        amplitudes = (
            events["amplitude"].to_numpy(dtype=float)
            if "amplitude" in events
            else np.ones_like(onsets)
        )
        frame_times = np.arange(n_scans, dtype=float) * self.tr
        regressor, _ = compute_regressor(
            exp_condition=(onsets, durations, amplitudes),
            hrf_model="spm",
            frame_times=frame_times,
        )
        stimulus = np.asarray(regressor[:, :1], dtype=np.float32)
        return self._match_stimulus_channels(self._zscore_channels(stimulus), n_scans)

    def _load_dense_stimulus(self, file_path: Path, target_n_scans: int) -> np.ndarray:
        if not file_path.exists():
            return np.zeros(
                (target_n_scans, int(config.STIMULUS_INPUT_CHANNELS)),
                dtype=np.float32,
            )

        suffix = file_path.suffix.lower()
        if suffix == ".mat":
            mat = scipy.io.loadmat(file_path)
            keys = [key for key in mat if not key.startswith("__")]
            if not keys:
                raise ValueError(f"No numeric variable found in {file_path}")
            raw_data = np.asarray(mat[keys[0]], dtype=np.float32)
        elif suffix == ".npy":
            raw_data = np.asarray(np.load(file_path), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dense stimulus format: {file_path}")

        if raw_data.ndim == 0:
            raw_data = raw_data.reshape(1, 1)
        elif raw_data.ndim == 1:
            raw_data = raw_data[:, None]
        elif raw_data.ndim == 2 and raw_data.shape[0] == 1 < raw_data.shape[1]:
            raw_data = raw_data.T

        if int(config.STIMULUS_INPUT_CHANNELS) == 1:
            raw_data = np.abs(raw_data)

        samples_per_tr = int(round(float(config.RAW_SAMPLING_RATE) * self.tr))
        if samples_per_tr <= 0:
            raise ValueError("RAW_SAMPLING_RATE * TR must be at least one sample")

        available_trs = raw_data.shape[0] // samples_per_tr
        if available_trs == 0:
            return np.zeros(
                (target_n_scans, int(config.STIMULUS_INPUT_CHANNELS)),
                dtype=np.float32,
            )

        truncated = raw_data[: available_trs * samples_per_tr]
        binned = truncated.reshape(
            (available_trs, samples_per_tr) + raw_data.shape[1:]
        ).mean(axis=1)
        binned = binned.reshape(available_trs, -1).astype(np.float32)

        should_convolve = bool(config.CONVOLVE_STIMULUS)
        if bool(config.USE_HEMODYNAMIC_HEAD):
            should_convolve = should_convolve and bool(
                config.ALLOW_STIMULUS_PRECONV_WITH_HEMO
            )

        if should_convolve:
            kernel = (
                canonical_cbv_response(self.tr, 30.0)
                if str(config.RESPONSE_FUNCTION).lower() == "cbv"
                else canonical_hrf_response(30.0, tr=self.tr)
            )
            convolved = np.zeros_like(binned)
            for channel in range(binned.shape[1]):
                convolved[:, channel] = scipy.signal.convolve(
                    binned[:, channel], kernel, mode="full"
                )[:available_trs]
            binned = convolved

        if binned.shape[0] < target_n_scans:
            binned = np.pad(
                binned,
                ((0, target_n_scans - binned.shape[0]), (0, 0)),
                mode="constant",
            )
        else:
            binned = binned[:target_n_scans]

        binned = self._zscore_channels(binned)
        return self._match_stimulus_channels(binned, target_n_scans)

    @staticmethod
    def _zscore_channels(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return np.nan_to_num((data - mean) / (std + 1e-6)).astype(np.float32)

    @staticmethod
    def _match_stimulus_channels(data: np.ndarray, target_length: int) -> np.ndarray:
        expected_channels = int(config.STIMULUS_INPUT_CHANNELS)
        data = np.asarray(data, dtype=np.float32).reshape(target_length, -1)
        if data.shape[1] < expected_channels:
            data = np.pad(
                data,
                ((0, 0), (0, expected_channels - data.shape[1])),
                mode="constant",
            )
        elif data.shape[1] > expected_channels:
            data = data[:, :expected_channels]
        return data.astype(np.float32)

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

    def _load_column_ids(self, mask_data: np.ndarray) -> torch.Tensor:
        columnar_path = Path(str(config.COLUMNAR_MASK_FILE))
        if not columnar_path.exists():
            return self.node_region_ids.clone()

        columnar_img = nib.load(str(columnar_path))
        columnar_img = resample_to_img(
            source_img=columnar_img,
            target_img=self.parcellation_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        columnar_data = np.asarray(columnar_img.get_fdata())
        column_ids: list[int] = []
        for region_id in self.region_ids:
            values = columnar_data[mask_data == region_id]
            values = values[np.isfinite(values) & (values > 0)].astype(int)
            column_ids.append(int(np.bincount(values).argmax()) if values.size else 0)
        return torch.tensor(column_ids, dtype=torch.long)

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