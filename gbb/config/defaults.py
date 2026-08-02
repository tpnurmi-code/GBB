"""Default values for the Glass-Box Brain configuration."""

from __future__ import annotations

import os
from pathlib import Path

from .schemas import (
    CfcConfig,
    Config,
    DataLoaderConfig,
    ExperimentConfig,
    FeatureExtractorConfig,
    GraphConfig,
    HemodynamicConfig,
    LossConfig,
    MaskerConfig,
    OptimizationConfig,
    PathConfig,
    SmoothnessConfig,
    StimulusConfig,
    TemporalStimulusEncoderConfig,
    TrainingConfig,
)

DEFAULT_DATA_DIR = Path(r"data")


def make_default_config(data_dir: str | Path | None = None) -> Config:
    """Construct a complete configuration with no placeholder values."""
    data_root = Path(
        data_dir
        if data_dir is not None
        else os.environ.get("GBB_DATA_DIR", str(DEFAULT_DATA_DIR))
    ).expanduser()

    results_dir = data_root / "results"
    log_dir = results_dir / "logs"

    return Config(
        experiment=ExperimentConfig(),
        paths=PathConfig(
            data_dir=data_root,
            results_dir=results_dir,
            log_dir=log_dir,
            checkpoint_path=results_dir / "feature_extractor_foundation.pth",
            mask_file=data_root / "group_roi_mask.nii",
            roi_mask_file=data_root / "group_roi_mask_10.nii",
            columnar_mask_file=data_root / "cortical_columns_7T.nii",
        ),
        masker=MaskerConfig(),
        stimulus=StimulusConfig(),
        hemodynamics=HemodynamicConfig(),
        dataloader=DataLoaderConfig(),
        training=TrainingConfig(),
        feature_extractor=FeatureExtractorConfig(),
        graph=GraphConfig(),
        cfc=CfcConfig(),
        temporal_stimulus=TemporalStimulusEncoderConfig(),
        losses=LossConfig(),
        smoothness=SmoothnessConfig(),
        optimization=OptimizationConfig(),
    ).finalize()


DEFAULT_CONFIG = make_default_config()
