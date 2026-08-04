from pathlib import Path

from gbb.synthetic import MechanisticSyntheticFMRI, SyntheticFMRIConfig


def test_quick_dataset_export_matches_gbb_layout(tmp_path: Path) -> None:
    config = SyntheticFMRIConfig.quick(tmp_path / "synthetic")
    config.overwrite = True
    result = MechanisticSyntheticFMRI(config).generate_dataset()

    assert (result.output_dir / "group_roi_mask.nii").exists()
    assert (result.output_dir / "group_roi_mask_10.nii").exists()
    assert (result.output_dir / "cortical_columns_7T.nii").exists()
    assert (result.output_dir / "mask_metadata.json").exists()
    assert (
        result.output_dir
        / "ground_truth"
        / "mechanistic_ground_truth.npz"
    ).exists()

    assert len(result.run_files) == config.num_subjects * config.num_runs

    first_subject = (
        result.output_dir
        / "synthetic_subject_001"
        / "NifTi"
    )

    assert (first_subject / "rfunctional_run1_events.tsv").exists()
    assert (first_subject / "rfunctional_run1_stim.mat").exists()
    assert (first_subject / "rfunctional_run1_ground_truth.npz").exists()
