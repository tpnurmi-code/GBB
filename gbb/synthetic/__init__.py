"""Mechanistic, privacy-safe synthetic fMRI generation for GBB."""

from .config import SyntheticFMRIConfig
from .generator import MechanisticSyntheticFMRI, SyntheticDatasetResult

__all__ = [
    "MechanisticSyntheticFMRI",
    "SyntheticDatasetResult",
    "SyntheticFMRIConfig",
]
