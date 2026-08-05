class StimulusLoader:
    def __init__(
        self,
        *,
        tr: float,
        settings: StimulusConfig,
        use_hemodynamic_head: bool,
    ) -> None:
        ...

    def load(
        self,
        run: dict[str, str],
        n_scans: int,
    ) -> np.ndarray:
    @staticmethod
    def _read_policy(
        name: str,
        allowed: set[str],
        *,
        default: str,
    ) -> str:
        value = str(getattr(config, name, default)).upper()
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"{name} must be one of {{{choices}}}; got {value!r}"
            )
        return value

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
                f"Stimulus from {source} has shape {stimulus.shape}; "
                f"expected {expected_shape}"
            )

        if not np.all(np.isfinite(stimulus)):
            raise ValueError(
                f"Stimulus from {source} contains NaN or infinite values"
            )

        if (
            self.require_nonzero_stimulus
            and not np.any(np.abs(stimulus) > 1e-8)
        ):
            return self._handle_stimulus_failure(
                f"Stimulus from {source} is entirely zero",
                n_scans,
            )

        return stimulus



    def _load_stimulus(
        self,
        item: dict[str, str],
        run_length: int,
    ) -> np.ndarray:
        if self.stimulus_mode == "EVENTS":
            return self._load_events_stimulus(
                item.get("events", ""),
                run_length,
            )

        if self.stimulus_mode == "DENSE":
            dense_path = self._resolve_dense_stimulus_path(item)
            return self._load_dense_stimulus(
                dense_path,
                run_length,
            )

        if self.stimulus_mode == "NONE":
            return self._zero_stimulus(run_length)

        raise RuntimeError(
            f"Unhandled stimulus mode: {self.stimulus_mode}"
        )

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
        item: dict[str, str],
    ) -> Path:
        extension = str(config.DENSE_STIMULUS_EXT)
        candidates: list[Path] = []

        events_text = str(item.get("events", "")).strip()
        if events_text.endswith("_events.tsv"):
            candidates.append(
                Path(
                    events_text[: -len("_events.tsv")]
                    + extension
                )
            )

        fmri_base = self._strip_nifti_suffix(
            Path(item["fmri"])
        )
        candidates.append(
            Path(str(fmri_base) + extension)
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return candidates[0]

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
        except Exception as exc:
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
        missing_columns = required_columns.difference(
            events.columns
        )

        if missing_columns:
            return self._handle_stimulus_failure(
                f"Events file {path} is missing required columns: "
                f"{sorted(missing_columns)}",
                n_scans,
            )

        if self.required_trial_type is not None:
            if "trial_type" not in events.columns:
                return self._handle_stimulus_failure(
                    f"Events file {path} has no 'trial_type' column, "
                    f"but REQUIRED_TRIAL_TYPE="
                    f"{self.required_trial_type!r}",
                    n_scans,
                )

            events = events[
                events["trial_type"]
                == self.required_trial_type
            ]

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
                f"Events file {path} contains non-numeric "
                f"stimulus values: {exc}",
                n_scans,
            )

        if not (
            np.all(np.isfinite(onsets))
            and np.all(np.isfinite(durations))
            and np.all(np.isfinite(amplitudes))
        ):
            return self._handle_stimulus_failure(
                f"Events file {path} contains NaN or "
                "infinite values",
                n_scans,
            )

        if np.any(durations < 0):
            return self._handle_stimulus_failure(
                f"Events file {path} contains negative durations",
                n_scans,
            )

        frame_times = np.arange(n_scans, dtype=float) * self.tr

        # Construct the event regressor without automatically imposing an HRF.
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

        stimulus = self._match_stimulus_channels(
            self._zscore_channels(stimulus),
            n_scans,
        )

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

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".mat":
                mat = scipy.io.loadmat(file_path)
                keys = [
                    key
                    for key in mat
                    if not key.startswith("__")
                ]

                if self.dense_stimulus_key is None:
                    if len(keys) != 1:
                        raise ValueError(
                            "Expected exactly one data variable in "
                            f"{file_path}; found {keys}"
                        )
                    key = keys[0]

                else:
                    key = str(self.dense_stimulus_key)
                    if key not in mat:
                        raise ValueError(
                            f"Dense stimulus key {key!r} was not "
                            f"found in {file_path}; available keys "
                            f"are {keys}"
                        )

                raw_data = np.asarray(
                    mat[key],
                    dtype=np.float32,
                )

            elif suffix == ".npy":
                raw_data = np.asarray(
                    np.load(file_path),
                    dtype=np.float32,
                )

            else:
                raise ValueError(
                    f"Unsupported dense stimulus format: "
                    f"{file_path}"
                )

        except (OSError, TypeError, ValueError) as exc:
            return self._handle_stimulus_failure(
                f"Could not load dense stimulus "
                f"{file_path}: {exc}",
                target_n_scans,
            )

        if raw_data.ndim == 0:
            raw_data = raw_data.reshape(1, 1)

        elif raw_data.ndim == 1:
            raw_data = raw_data[:, None]

        elif (
            raw_data.ndim == 2
            and raw_data.shape[0] == 1
            and raw_data.shape[1] > 1
        ):
            raw_data = raw_data.T

        if not np.all(np.isfinite(raw_data)):
            return self._handle_stimulus_failure(
                f"Dense stimulus {file_path} contains NaN "
                "or infinite values",
                target_n_scans,
            )

        if self.stimulus_channels == 1:
            raw_data = np.abs(raw_data)

        samples_per_tr = int(
            round(
                float(config.RAW_SAMPLING_RATE)
                * self.tr
            )
        )

        if samples_per_tr <= 0:
            raise ValueError(
                "RAW_SAMPLING_RATE * TR must be at least "
                "one sample"
            )

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
                f"Dense stimulus {file_path} has "
                f"{remainder} trailing samples that do not "
                "form a complete TR"
            )

        truncated = raw_data[
            : available_trs * samples_per_tr
        ]

        binned = truncated.reshape(
            (available_trs, samples_per_tr)
            + raw_data.shape[1:]
        ).mean(axis=1)

        binned = binned.reshape(
            available_trs,
            -1,
        ).astype(np.float32)

        should_convolve = bool(
            config.CONVOLVE_STIMULUS
        )

        if self._should_convolve_stimulus():
            kernel = self._stimulus_response_kernel()

            convolved = np.zeros_like(binned)

            for channel in range(binned.shape[1]):
                convolved[:, channel] = (
                    scipy.signal.convolve(
                        binned[:, channel],
                        kernel,
                        mode="full",
                    )[:available_trs]
                )

            binned = convolved

        if binned.shape[0] != target_n_scans:
            self._handle_shape_mismatch(
                f"Dense stimulus {file_path} produces "
                f"{binned.shape[0]} TRs; the fMRI run "
                f"requires {target_n_scans}"
            )

            if binned.shape[0] < target_n_scans:
                binned = np.pad(
                    binned,
                    (
                        (
                            0,
                            target_n_scans
                            - binned.shape[0],
                        ),
                        (0, 0),
                    ),
                    mode="constant",
                )

            else:
                binned = binned[:target_n_scans]

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

    @staticmethod
    def _zscore_channels(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return np.nan_to_num((data - mean) / (std + 1e-6)).astype(np.float32)



    def _match_stimulus_channels(
        self,
        data: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        data = np.asarray(
            data,
            dtype=np.float32,
        )

        if data.ndim == 1:
            data = data[:, None]

        if data.ndim != 2:
            raise ValueError(
                "Stimulus must be a two-dimensional "
                f"time-by-channel array; got {data.shape}"
            )

        if data.shape[0] != target_length:
            self._handle_shape_mismatch(
                f"Stimulus has {data.shape[0]} time points; "
                f"expected {target_length}"
            )

            if data.shape[0] < target_length:
                data = np.pad(
                    data,
                    (
                        (
                            0,
                            target_length - data.shape[0],
                        ),
                        (0, 0),
                    ),
                    mode="constant",
                )
            else:
                data = data[:target_length]

        if data.shape[1] != self.stimulus_channels:
            self._handle_shape_mismatch(
                f"Stimulus has {data.shape[1]} channels; "
                f"expected {self.stimulus_channels}"
            )

            if data.shape[1] < self.stimulus_channels:
                data = np.pad(
                    data,
                    (
                        (0, 0),
                        (
                            0,
                            self.stimulus_channels
                            - data.shape[1],
                        ),
                    ),
                    mode="constant",
                )
            else:
                data = data[
                    :,
                    : self.stimulus_channels,
                ]

        return data.astype(np.float32)

