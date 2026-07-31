"""File discovery utilities."""

from __future__ import annotations

from pathlib import Path


def get_subject_files(data_dir, num_runs: int = 2) -> list[dict[str, str]]:
    """Discover fMRI runs without performing the train/test split."""
    data_root = Path(data_dir)
    mask_path = data_root / "group_roi_mask.nii"
    data_list: list[dict[str, str]] = []

    for subject_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        subject_id = subject_dir.name
        nifti_dir = subject_dir / "NifTi"
        for run_number in range(1, num_runs + 1):
            candidates = [
                nifti_dir / f"rfunctional_run{run_number}.nii",
                nifti_dir / f"rfunctional_run{run_number}.nii.gz",
            ]
            fmri_path = next((path for path in candidates if path.exists()), None)
            if fmri_path is None:
                continue

            stem = str(fmri_path)
            if stem.endswith(".nii.gz"):
                events_path = Path(stem[:-7] + "_events.tsv")
            else:
                events_path = fmri_path.with_suffix("").with_name(fmri_path.stem + "_events.tsv")

            data_list.append(
                {
                    "id": f"{subject_id}_run{run_number}",
                    "subject_id": subject_id,
                    "fmri": str(fmri_path),
                    "mask": str(mask_path),
                    "events": str(events_path),
                }
            )
    return data_list