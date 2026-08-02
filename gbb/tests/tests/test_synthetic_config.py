from pathlib import Path

import pytest

from gbb.synthetic import SyntheticFMRIConfig


def test_quick_config_is_valid(tmp_path: Path) -> None:
    config = SyntheticFMRIConfig.quick(tmp_path / "synthetic")
    config.validate()
    assert config.num_nodes == config.num_columns * len(config.cortical_layers)
    assert config.neural_steps > config.n_timepoints


def test_invalid_tau_range_is_rejected() -> None:
    config = SyntheticFMRIConfig(tau_min_s=2.0, tau_max_s=1.0)
    with pytest.raises(ValueError, match="tau_max_s"):
        config.validate()
