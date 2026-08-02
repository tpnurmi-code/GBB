"""High-level mechanistic synthetic fMRI dataset generator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .anatomy import SyntheticAnatomy, build_synthetic_anatomy
from .config import SyntheticFMRIConfig
from .dynamics import (
    NeuralGroundTruth,
    build_neural_ground_truth,
    make_block_stimulus,
    simulate_neural_dynamics,
)
from .export import (
    prepare_output_directory,
    write_dataset_description,
    write_dense_stimulus_mat,
    write_events_tsv,
    write_fmri_nifti,
    write_ground_truth,
    write_masks_and_metadata,
    write_node_timeseries_npz,
)
from .hemodynamics import (
    HemodynamicGroundTruth,
    build_hemodynamic_ground_truth,
    observe_fmri,
)
from .network import GroundTruthNetwork, build_ground_truth_network
from .noise import add_measurement_noise, build_measurement_noise


@dataclass(slots=True)
class SyntheticRunResult:
    subject_id: str
    run_number: int
    observed_fmri: np.ndarray
    noiseless_fmri: np.ndarray
    neural_activity: np.ndarray
    stimulus: np.ndarray
    events: list[dict[str, float | str]]
    diagnostics: dict[str, np.ndarray]
    subject_tau_scale: np.ndarray
    subject_connectivity_scale: np.ndarray
    subject_hemodynamic_scale: np.ndarray


@dataclass(slots=True)
class SyntheticDatasetResult:
    output_dir: Path
    anatomy: SyntheticAnatomy
    network: GroundTruthNetwork
    neural_ground_truth: NeuralGroundTruth
    hemodynamic_ground_truth: HemodynamicGroundTruth
    run_files: tuple[Path, ...]


class MechanisticSyntheticFMRI:
    """Generate a privacy-safe fMRI dataset with known mechanistic parameters."""

    def __init__(self, config: SyntheticFMRIConfig | None = None) -> None:
        self.config = config or SyntheticFMRIConfig()
        self.config.validate()
        self.anatomy = build_synthetic_anatomy(self.config)
        self.network = build_ground_truth_network(self.config, self.anatomy)
        self.neural_ground_truth = build_neural_ground_truth(self.config, self.anatomy)
        self.hemodynamic_ground_truth = build_hemodynamic_ground_truth(
            self.config, self.anatomy
        )

    def _subject_scales(
        self, subject_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed + 5003 + subject_index)
        tau_scale = rng.lognormal(
            mean=0.0,
            sigma=self.config.subject_tau_sd_fraction,
            size=self.anatomy.num_nodes,
        )
        connectivity_scale = rng.lognormal(
            mean=0.0,
            sigma=self.config.subject_connectivity_sd_fraction,
            size=self.network.num_edges,
        )
        hemodynamic_scale = rng.lognormal(
            mean=0.0,
            sigma=self.config.subject_hrf_sd_fraction,
            size=self.anatomy.num_nodes,
        )
        return tau_scale, connectivity_scale, hemodynamic_scale

    def simulate_run(self, subject_index: int, run_number: int) -> SyntheticRunResult:
        """Simulate one subject/run entirely in memory."""
        if subject_index < 0:
            raise ValueError("subject_index must be non-negative")
        if run_number < 1:
            raise ValueError("run_number must start at one")

        subject_id = f"synthetic_subject_{subject_index + 1:03d}"
        base_seed = self.config.seed + subject_index * 100_000 + run_number * 1_000
        rng = np.random.default_rng(base_seed)
        tau_scale, connectivity_scale, hemodynamic_scale = self._subject_scales(subject_index)
        stimulus, events = make_block_stimulus(self.config, rng)

        neural_activity, diagnostics = simulate_neural_dynamics(
            self.config,
            self.anatomy,
            self.network,
            self.neural_ground_truth,
            stimulus,
            seed=base_seed + 17,
            tau_scale=tau_scale,
            connectivity_scale=connectivity_scale,
        )
        subject_hemodynamics = build_hemodynamic_ground_truth(
            self.config,
            self.anatomy,
            seed_offset=subject_index * 100 + run_number,
            subject_scale=hemodynamic_scale,
        )
        noiseless = observe_fmri(self.config, neural_activity, subject_hemodynamics)
        run_scale = float(
            rng.lognormal(mean=0.0, sigma=self.config.run_noise_scale_sd_fraction)
        )
        noise = build_measurement_noise(
            self.config,
            self.anatomy,
            seed=base_seed + 31,
            run_scale=run_scale,
        )
        observed = add_measurement_noise(noiseless, noise)

        # Preserve a realistic range while avoiding a few nuisance spikes from
        # dominating small tutorial datasets.
        observed = np.clip(observed, -8.0, 8.0).astype(np.float32)
        noise_dict = {
            "temporal_ar": noise.temporal_ar,
            "spatial_measurement": noise.spatial_measurement,
            "global_signal": noise.global_signal,
            "drift": noise.drift,
            "physiological": noise.physiological,
            "motion": noise.motion,
        }
        diagnostics = dict(diagnostics)
        diagnostics["noise_total"] = noise.total.astype(np.float32)
        diagnostics["noise_temporal_ar"] = noise.temporal_ar
        diagnostics["noise_spatial_measurement"] = noise.spatial_measurement
        diagnostics["noise_global_signal"] = noise.global_signal
        diagnostics["noise_drift"] = noise.drift
        diagnostics["noise_physiological"] = noise.physiological
        diagnostics["noise_motion"] = noise.motion
        diagnostics["_noise_component_names"] = np.asarray(list(noise_dict), dtype="U32")

        return SyntheticRunResult(
            subject_id=subject_id,
            run_number=run_number,
            observed_fmri=observed,
            noiseless_fmri=noiseless.astype(np.float32),
            neural_activity=neural_activity,
            stimulus=stimulus.astype(np.float32),
            events=events,
            diagnostics=diagnostics,
            subject_tau_scale=tau_scale.astype(np.float32),
            subject_connectivity_scale=connectivity_scale.astype(np.float32),
            subject_hemodynamic_scale=hemodynamic_scale.astype(np.float32),
        )

    def generate_dataset(self) -> SyntheticDatasetResult:
        """Write a complete GBB-compatible synthetic dataset to disk."""
        output_dir = prepare_output_directory(self.config)
        write_dataset_description(output_dir, self.config)
        label_mask, affine, voxels_by_node = write_masks_and_metadata(
            self.config, self.anatomy, output_dir
        )
        write_ground_truth(
            output_dir,
            self.config,
            self.anatomy,
            self.network,
            self.neural_ground_truth,
            self.hemodynamic_ground_truth,
        )

        run_files: list[Path] = []
        subject_manifest: dict[str, dict[str, object]] = {}
        for subject_index in range(self.config.num_subjects):
            subject_id = f"synthetic_subject_{subject_index + 1:03d}"
            nifti_dir = output_dir / subject_id / "NifTi"
            nifti_dir.mkdir(parents=True, exist_ok=True)
            subject_manifest[subject_id] = {"runs": []}

            for run_number in range(1, self.config.num_runs + 1):
                result = self.simulate_run(subject_index, run_number)
                stem = nifti_dir / f"rfunctional_run{run_number}"
                fmri_path = write_fmri_nifti(
                    self.config,
                    result.observed_fmri,
                    label_mask,
                    affine,
                    voxels_by_node,
                    stem,
                    seed=self.config.seed + subject_index * 10_000 + run_number,
                )
                run_files.append(fmri_path)

                if self.config.save_events_tsv:
                    write_events_tsv(result.events, Path(str(stem) + "_events.tsv"))
                if self.config.save_dense_stimulus_mat:
                    write_dense_stimulus_mat(
                        self.config,
                        result.stimulus,
                        Path(str(stem) + "_stim.mat"),
                        seed=self.config.seed + subject_index * 10_000 + run_number + 91,
                    )
                if self.config.save_node_timeseries_npz:
                    noise_components = {
                        key.removeprefix("noise_"): value
                        for key, value in result.diagnostics.items()
                        if key.startswith("noise_") and key != "noise_total"
                    }
                    write_node_timeseries_npz(
                        Path(str(stem) + "_ground_truth.npz"),
                        observed=result.observed_fmri,
                        noiseless=result.noiseless_fmri,
                        neural=result.neural_activity,
                        stimulus=result.stimulus,
                        diagnostics={
                            key: value
                            for key, value in result.diagnostics.items()
                            if not key.startswith("noise_") and not key.startswith("_noise")
                        },
                        noise_components=noise_components,
                    )

                run_meta = {
                    "run_number": run_number,
                    "fmri": str(fmri_path.relative_to(output_dir)),
                    "events": str(Path(str(stem) + "_events.tsv").relative_to(output_dir)),
                    "dense_stimulus": str(Path(str(stem) + "_stim.mat").relative_to(output_dir)),
                    "subject_tau_scale_mean": float(result.subject_tau_scale.mean()),
                    "subject_connectivity_scale_mean": float(
                        result.subject_connectivity_scale.mean()
                    ),
                    "subject_hemodynamic_scale_mean": float(
                        result.subject_hemodynamic_scale.mean()
                    ),
                }
                subject_manifest[subject_id]["runs"].append(run_meta)  # type: ignore[index]

        (output_dir / "subjects_manifest.json").write_text(
            json.dumps(subject_manifest, indent=2), encoding="utf-8"
        )
        return SyntheticDatasetResult(
            output_dir=output_dir,
            anatomy=self.anatomy,
            network=self.network,
            neural_ground_truth=self.neural_ground_truth,
            hemodynamic_ground_truth=self.hemodynamic_ground_truth,
            run_files=tuple(run_files),
        )
