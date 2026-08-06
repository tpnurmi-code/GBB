"""Strict event and dense stimulus loading for fMRI runs."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from nilearn.glm.first_level import compute_regressor

from gbb.config.schemas import StimulusConfig
from gbb.data.hrf import canonical_cbv_response, canonical_hrf_response


class StimulusLoader:
    """Load and validate event-based or dense stimulus inputs."""

    def __init__(
        self,
        *,
        tr: float,
        settings: StimulusConfig,
        use_hemodynamic_head: bool,
    ) -> None:
        self.tr = float(tr)
        if self.tr <= 0:
            raise ValueError("tr must be positive")

        self.settings = settings
        self.use_hemodynamic_head = bool(use_hemodynamic_head)

        self.stimulus_mode = self._read_policy(
            "mode",
            settings.mode,
            {"EVENTS", "DENSE", "NONE"},
        )
        self.missing_stimulus_policy = self._read_policy(
            "missing_stimulus_policy",
            settings.missing_stimulus_policy,
            {"ERROR", "WARN", "ZEROS"},
        )
        self.stimulus_shape_policy = self._read_policy(
            "stimulus_shape_policy",
            settings.stimulus_shape_policy,
            {"ERROR", "WARN", "COERCE"},
        )

        self.stimulus_channels = int(settings.input_channels)
        if self.stimulus_channels <= 0:
            raise ValueError("Stimulus input_channels must be positive")

        self.require_nonzero_stimulus = bool(settings.require_nonzero_stimulus)
        self.required_trial_type = settings.required_trial_type
        self.dense_stimulus_key = settings.dense_stimulus_key
        self.dense_stimulus_ext = str(settings.dense_stimulus_ext)
        self.raw_sampling_rate = float(settings.raw_sampling_rate)
        if self.raw_sampling_rate <= 0:
            raise ValueError("Stimulus raw_sampling_rate must be positive")

        self.response_function = str(settings.response_function).lower()
        if self.response_function not in {"hrf", "cbv", "uniform"}:
            raise ValueError(
                "response_function must be 'hrf', 'cbv', or 'uniform'; "
                f"got {self.response_function!r}"
            )

        self.convolve_stimulus = bool(settings.convolve_stimulus)
        self.allow_preconvolution_with_hemodynamic_head = bool(
            settings.allow_stimulus_preconv_with_hemo
        )

    @staticmethod
    def _read_policy(
        name: str,
        value: object,
        allowed: set[str],
    ) -> str:
        normalized = str(value).upper()
        if normalized not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"{name} must be one of {{{choices}}}; got {normalized!r}")
        return normalized

    def load(
        self,
        run: dict[str, str],
        n_scans: int,
    ) -> np.ndarray:
        """Load one run's stimulus with shape ``(n_scans, channels)``."""
        if n_scans <= 0:
            raise ValueError("n_scans must be positive")

        if self.stimulus_mode == "EVENTS":
            return self._load_events_stimulus(
                run.get("events", ""),
                n_scans,
            )
        if self.stimulus_mode == "DENSE":
            return self._load_dense_stimulus(
                self._resolve_dense_stimulus_path(run),
                n_scans,
            )
        if self.stimulus_mode == "NONE":
            return self._zero_stimulus(n_scans)

        raise RuntimeError(f"Unhandled stimulus mode: {self.stimulus_mode}")

    def _zero_stimulus(self, n_scans: int) -> np.ndarray:
        return np.zeros(
            (n_scans, self.stimulus_channels),
            dtype=np.float32,
        )

    def _handle_stimulus_failure(
        self,
        message: str,
        n_scans: int,
        *,
        exception_type: type[Exception] = ValueError,
    ) -> np.ndarray:
        if self.missing_stimulus_policy == "ERROR":
            raise exception_type(message)

        if self.missing_stimulus_policy == "WARN":
            warnings.warn(
                f"{message} Substituting an all-zero stimulus.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self._zero_stimulus(n_scans)

    def _handle_shape_mismatch(self, message: str) -> None:
        if self.stimulus_shape_policy == "ERROR":
            raise ValueError(message)

        if self.stimulus_shape_policy == "WARN":
            warnings.warn(
                f"{message} Coercing the stimulus to the configured shape.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _validate_loaded_stimulus(
        self,
        stimulus: np.ndarray,
        n_scans: int,
        source: str,
    ) -> np.ndarray:
        stimulus = np.asarray(stimulus, dtype=np.float32)
        expected_shape = (n_scans, self.stimulus_channels)

        if stimulus.shape != expected_shape:
            raise ValueError(
                f"Stimulus from {source} has shape {stimulus.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(stimulus)):
            raise ValueError(f"Stimulus from {source} contains NaN or infinite values")
        if self.require_nonzero_stimulus and not np.any(np.abs(stimulus) > 1e-8):
            return self._handle_stimulus_failure(
                f"Stimulus from {source} is entirely zero",
                n_scans,
            )

        return stimulus

    @staticmethod
    def _strip_nifti_suffix(path: Path) -> Path:
        text = str(path)
        if text.endswith(".nii.gz"):
            return Path(text[:-7])
        if text.endswith(".nii"):
            return Path(text[:-4])
        return path.with_suffix("")

    def _resolve_dense_stimulus_path(
        self,
        run: dict[str, str],
    ) -> Path:
        if "fmri" not in run:
            raise KeyError("Dense stimulus path resolution requires an 'fmri' path")

        candidates: list[Path] = []
        events_text = str(run.get("events", "")).strip()
        if events_text.endswith("_events.tsv"):
            candidates.append(Path(events_text[: -len("_events.tsv")] + self.dense_stimulus_ext))

        fmri_base = self._strip_nifti_suffix(Path(run["fmri"]))
        candidates.append(Path(str(fmri_base) + self.dense_stimulus_ext))

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return candidates[0]

    def _should_convolve_stimulus(self) -> bool:
        if not self.convolve_stimulus:
            return False
        if self.use_hemodynamic_head:
            return self.allow_preconvolution_with_hemodynamic_head
        return True

    def _stimulus_response_kernel(self) -> np.ndarray:
        if self.response_function == "cbv":
            return canonical_cbv_response(self.tr, 30.0)
        if self.response_function == "hrf":
            return canonical_hrf_response(30.0, tr=self.tr)
        if self.response_function == "uniform":
            return np.ones(1, dtype=np.float32)

        raise RuntimeError(f"Unhandled response function: {self.response_function}")

    def _load_events_stimulus(
        self,
        events_path: str | os.PathLike[str],
        n_scans: int,
    ) -> np.ndarray:
        path_text = str(events_path).strip()
        if not path_text:
            return self._handle_stimulus_failure(
                "No events file path was supplied",
                n_scans,
                exception_type=FileNotFoundError,
            )

        path = Path(path_text)
        if not path.is_file():
            return self._handle_stimulus_failure(
                f"Events file not found: {path}",
                n_scans,
                exception_type=FileNotFoundError,
            )

        try:
            events = pd.read_csv(path, sep="\t")
        except (
            OSError,
            UnicodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as exc:
            return self._handle_stimulus_failure(
                f"Could not read events file {path}: {exc}",
                n_scans,
            )

        if events.empty:
            return self._handle_stimulus_failure(
                f"Events file is empty: {path}",
                n_scans,
            )

        required_columns = {"onset", "duration"}
        missing_columns = required_columns.difference(events.columns)
        if missing_columns:
            return self._handle_stimulus_failure(
                f"Events file {path} is missing required columns: {sorted(missing_columns)}",
                n_scans,
            )

        if self.required_trial_type is not None:
            if "trial_type" not in events.columns:
                return self._handle_stimulus_failure(
                    f"Events file {path} has no 'trial_type' column, "
                    f"but required_trial_type={self.required_trial_type!r}",
                    n_scans,
                )

            events = events[events["trial_type"] == self.required_trial_type]
            if events.empty:
                return self._handle_stimulus_failure(
                    f"Events file {path} contains no rows with "
                    f"trial_type={self.required_trial_type!r}",
                    n_scans,
                )

        try:
            onsets = pd.to_numeric(
                events["onset"],
                errors="raise",
            ).to_numpy(dtype=float)
            durations = pd.to_numeric(
                events["duration"],
                errors="raise",
            ).to_numpy(dtype=float)
            amplitudes = (
                pd.to_numeric(
                    events["amplitude"],
                    errors="raise",
                ).to_numpy(dtype=float)
                if "amplitude" in events.columns
                else np.ones_like(onsets)
            )
        except (TypeError, ValueError) as exc:
            return self._handle_stimulus_failure(
                f"Events file {path} contains non-numeric stimulus values: {exc}",
                n_scans,
            )

        if not (
            np.all(np.isfinite(onsets))
            and np.all(np.isfinite(durations))
            and np.all(np.isfinite(amplitudes))
        ):
            return self._handle_stimulus_failure(
                f"Events file {path} contains NaN or infinite values",
                n_scans,
            )
        if np.any(durations < 0):
            return self._handle_stimulus_failure(
                f"Events file {path} contains negative durations",
                n_scans,
            )

        frame_times = np.arange(n_scans, dtype=float) * self.tr
        regressor, _ = compute_regressor(
            exp_condition=(onsets, durations, amplitudes),
            hrf_model=None,
            frame_times=frame_times,
        )
        stimulus = np.asarray(regressor[:, :1], dtype=np.float32)

        if self._should_convolve_stimulus():
            kernel = self._stimulus_response_kernel()
            stimulus[:, 0] = scipy.signal.convolve(
                stimulus[:, 0],
                kernel,
                mode="full",
            )[:n_scans]

        stimulus = self._zscore_channels(stimulus)
        stimulus = self._match_stimulus_channels(stimulus, n_scans)
        return self._validate_loaded_stimulus(
            stimulus,
            n_scans,
            str(path),
        )

    def _load_dense_stimulus(
        self,
        file_path: Path,
        target_n_scans: int,
    ) -> np.ndarray:
        if not file_path.is_file():
            return self._handle_stimulus_failure(
                f"Dense stimulus file not found: {file_path}",
                target_n_scans,
                exception_type=FileNotFoundError,
            )

        try:
            raw_data = self._read_dense_array(file_path)
        except (OSError, TypeError, ValueError, NotImplementedError) as exc:
            return self._handle_stimulus_failure(
                f"Could not load dense stimulus {file_path}: {exc}",
                target_n_scans,
            )

        if raw_data.ndim == 0:
            raw_data = raw_data.reshape(1, 1)
        elif raw_data.ndim == 1:
            raw_data = raw_data[:, None]
        elif raw_data.ndim == 2 and raw_data.shape[0] == 1 and raw_data.shape[1] > 1:
            raw_data = raw_data.T

        if not np.all(np.isfinite(raw_data)):
            return self._handle_stimulus_failure(
                f"Dense stimulus {file_path} contains NaN or infinite values",
                target_n_scans,
            )

        if self.stimulus_channels == 1:
            raw_data = np.abs(raw_data)

        samples_per_tr = int(round(self.raw_sampling_rate * self.tr))
        if samples_per_tr <= 0:
            raise ValueError("raw_sampling_rate * tr must be at least one sample")

        available_trs, remainder = divmod(
            raw_data.shape[0],
            samples_per_tr,
        )
        if available_trs == 0:
            return self._handle_stimulus_failure(
                f"Dense stimulus {file_path} has only "
                f"{raw_data.shape[0]} samples; at least "
                f"{samples_per_tr} are required for one TR",
                target_n_scans,
            )
        if remainder:
            self._handle_shape_mismatch(
                f"Dense stimulus {file_path} has {remainder} trailing "
                "samples that do not form a complete TR"
            )

        truncated = raw_data[: available_trs * samples_per_tr]
        binned = truncated.reshape((available_trs, samples_per_tr) + raw_data.shape[1:]).mean(
            axis=1
        )
        binned = binned.reshape(available_trs, -1).astype(np.float32)

        if self._should_convolve_stimulus():
            kernel = self._stimulus_response_kernel()
            convolved = np.zeros_like(binned)
            for channel in range(binned.shape[1]):
                convolved[:, channel] = scipy.signal.convolve(
                    binned[:, channel],
                    kernel,
                    mode="full",
                )[:available_trs]
            binned = convolved

        if binned.shape[0] != target_n_scans:
            self._handle_shape_mismatch(
                f"Dense stimulus {file_path} produces "
                f"{binned.shape[0]} TRs; the fMRI run requires "
                f"{target_n_scans}"
            )
            binned = self._coerce_length(binned, target_n_scans)

        binned = self._zscore_channels(binned)
        binned = self._match_stimulus_channels(
            binned,
            target_n_scans,
        )
        return self._validate_loaded_stimulus(
            binned,
            target_n_scans,
            str(file_path),
        )

    def _read_dense_array(self, file_path: Path) -> np.ndarray:
        suffix = file_path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(
                np.load(file_path, allow_pickle=False),
                dtype=np.float32,
            )
        if suffix != ".mat":
            raise ValueError(f"Unsupported dense stimulus format: {file_path}")

        mat = scipy.io.loadmat(file_path)
        keys = [key for key in mat if not key.startswith("__")]

        if self.dense_stimulus_key is None:
            if len(keys) != 1:
                raise ValueError(f"Expected exactly one data variable in {file_path}; found {keys}")
            key = keys[0]
        else:
            key = str(self.dense_stimulus_key)
            if key not in mat:
                raise ValueError(
                    f"Dense stimulus key {key!r} was not found in "
                    f"{file_path}; available keys are {keys}"
                )

        return np.asarray(mat[key], dtype=np.float32)

    @staticmethod
    def _zscore_channels(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Stimulus must be two-dimensional before z-scoring; got {data.shape}")

        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return ((data - mean) / (std + 1e-6)).astype(np.float32)

    @staticmethod
    def _coerce_length(
        data: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        if data.shape[0] < target_length:
            return np.pad(
                data,
                ((0, target_length - data.shape[0]), (0, 0)),
                mode="constant",
            )
        return data[:target_length]

    def _match_stimulus_channels(
        self,
        data: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError(
                f"Stimulus must be a two-dimensional time-by-channel array; got {data.shape}"
            )

        if data.shape[0] != target_length:
            self._handle_shape_mismatch(
                f"Stimulus has {data.shape[0]} time points; expected {target_length}"
            )
            data = self._coerce_length(data, target_length)

        if data.shape[1] != self.stimulus_channels:
            self._handle_shape_mismatch(
                f"Stimulus has {data.shape[1]} channels; expected {self.stimulus_channels}"
            )
            if data.shape[1] < self.stimulus_channels:
                data = np.pad(
                    data,
                    (
                        (0, 0),
                        (0, self.stimulus_channels - data.shape[1]),
                    ),
                    mode="constant",
                )
            else:
                data = data[:, : self.stimulus_channels]

        return data.astype(np.float32)
