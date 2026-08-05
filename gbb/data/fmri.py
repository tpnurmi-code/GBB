def load_fmri_run(
    *,
    fmri_path: Path,
    masker: NiftiLabelsMasker,
    parcellation_img,
    expected_nodes: int,
) -> np.ndarray:
    ...