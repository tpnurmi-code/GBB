import numpy as np

from gbb.synthetic import MechanisticSyntheticFMRI, SyntheticFMRIConfig


def lag_one_autocorrelation(data: np.ndarray) -> float:
    x = data[:-1].reshape(-1)
    y = data[1:].reshape(-1)
    return float(np.corrcoef(x, y)[0, 1])


def test_in_memory_run_shapes_and_finiteness() -> None:
    config = SyntheticFMRIConfig.quick()
    generator = MechanisticSyntheticFMRI(config)
    result = generator.simulate_run(subject_index=0, run_number=1)

    assert result.observed_fmri.shape == (config.n_timepoints, config.num_nodes)
    assert result.noiseless_fmri.shape == result.observed_fmri.shape
    assert result.neural_activity.shape == (config.neural_steps, config.num_nodes)
    assert result.stimulus.shape == (config.neural_steps,)
    assert np.all(np.isfinite(result.observed_fmri))
    assert np.all(np.isfinite(result.neural_activity))
    assert result.stimulus.max() > 0
    assert len(result.events) >= 1


def test_observed_signal_has_fmri_like_temporal_structure() -> None:
    result = MechanisticSyntheticFMRI(SyntheticFMRIConfig.quick()).simulate_run(0, 1)
    assert lag_one_autocorrelation(result.observed_fmri) > 0.10
    assert result.observed_fmri.std() > 0.20


def test_generation_is_reproducible_for_same_seed() -> None:
    config_a = SyntheticFMRIConfig.quick()
    config_b = SyntheticFMRIConfig.quick()
    run_a = MechanisticSyntheticFMRI(config_a).simulate_run(0, 1)
    run_b = MechanisticSyntheticFMRI(config_b).simulate_run(0, 1)
    np.testing.assert_allclose(run_a.observed_fmri, run_b.observed_fmri)


def test_subjects_have_related_but_nonidentical_parameters() -> None:
    generator = MechanisticSyntheticFMRI(SyntheticFMRIConfig.quick())
    run_a = generator.simulate_run(0, 1)
    run_b = generator.simulate_run(1, 1)
    assert not np.allclose(run_a.subject_tau_scale, run_b.subject_tau_scale)
    assert not np.allclose(run_a.observed_fmri, run_b.observed_fmri)
