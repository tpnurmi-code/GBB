"""Mechanistic, privacy-safe synthetic fMRI generation for GBB."""

from .config import SyntheticFMRIConfig
from .generator import MechanisticSyntheticFMRI, SyntheticDatasetResult
from .profiles import GroundTruthProfile, load_ground_truth_profile

__all__ = [
    "GroundTruthProfile",
    "MechanisticSyntheticFMRI",
    "SyntheticDatasetResult",
    "SyntheticFMRIConfig",
    "load_ground_truth_profile",
]
