from pathlib import Path

from gbb.synthetic import MechanisticSyntheticFMRI, SyntheticFMRIConfig


config = SyntheticFMRIConfig(
    # Output
    output_dir=Path("synthetic_20_subject_cbv"),
    overwrite=True,
    seed=42,

    # Dataset size
    num_subjects=20,
    num_runs=2,
    num_columns=12,

    # Recording
    n_timepoints=240,
    tr=2.0,
    neural_dt=0.10,
    response_kind="cbv",

    # Event design
    first_block_onset_s=15.0,
    block_duration_s=10.0,
    inter_block_interval_s=20.0,
    stimulus_amplitude=1.0,
    stimulus_jitter_s=1.0,

    # Subject variability
    subject_tau_sd_fraction=0.06,
    subject_connectivity_sd_fraction=0.05,
    subject_hrf_sd_fraction=0.08,

    # Noise
    neural_noise_sd=0.08,
    measurement_noise_sd=0.45,
    global_signal_sd=0.18,
    drift_sd=0.20,
    physiological_sd=0.08,
)

config.validate()

result = MechanisticSyntheticFMRI(config).generate_dataset()

print("Dataset generated successfully")
print("Directory:", result.output_dir)
print("Subjects:", config.num_subjects)
print("Runs per subject:", config.num_runs)
print("Duration per run:", config.duration_s, "seconds")
print("Nodes:", result.anatomy.num_nodes)
print("Response:", config.response_kind.upper())