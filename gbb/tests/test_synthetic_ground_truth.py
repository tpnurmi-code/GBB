import numpy as np

from gbb.synthetic import MechanisticSyntheticFMRI, SyntheticFMRIConfig


def make_generator() -> MechanisticSyntheticFMRI:
    return MechanisticSyntheticFMRI(SyntheticFMRIConfig.quick())


def test_tau_tracks_known_hierarchy() -> None:
    generator = make_generator()
    correlation = np.corrcoef(
        generator.neural_ground_truth.tau_seconds,
        generator.anatomy.hierarchy,
    )[0, 1]
    assert correlation > 0.75


def test_intrinsic_drive_is_signed() -> None:
    drive = make_generator().neural_ground_truth.intrinsic_drive
    assert np.any(drive > 0)
    assert np.any(drive < 0)


def test_network_is_sparse_directed_and_multichannel() -> None:
    generator = make_generator()
    network = generator.network
    assert 0.01 < network.density < 0.30
    assert set(np.unique(network.channel)) == {0, 1, 2}
    assert np.all(network.signed_weight[network.channel == 0] > 0)
    assert np.all(network.signed_weight[network.channel == 1] < 0)
    assert np.all(network.delay_steps >= 1)
    assert np.ptp(network.delay_seconds) > 0

    combined = np.any(network.adjacency_by_channel > 0, axis=0)
    assert np.any(combined != combined.T)


def test_fastkan_parameters_are_edge_specific() -> None:
    network = make_generator().network
    assert network.rbf_centers.shape == network.rbf_coefficients.shape
    assert network.rbf_widths.shape == network.rbf_coefficients.shape
    assert network.rbf_coefficients.shape[0] == network.num_edges
    assert np.std(network.rbf_coefficients, axis=0).mean() > 0


def test_hemodynamic_parameters_vary_by_region() -> None:
    hemo = make_generator().hemodynamic_ground_truth
    assert np.ptp(hemo.time_to_peak_s) > 0.25
    assert np.ptp(hemo.amplitude) > 0.10
    assert np.all(np.isfinite(hemo.kernels))


def test_toy_label_volume_contains_every_node() -> None:
    from gbb.synthetic.export import build_label_volumes

    generator = make_generator()
    label_mask, coarse_mask, column_mask, affine, voxels = build_label_volumes(
        generator.config, generator.anatomy
    )
    ids = np.unique(label_mask)
    ids = ids[ids > 0]
    assert ids.size == generator.anatomy.num_nodes
    assert len(voxels) == generator.anatomy.num_nodes
    assert coarse_mask.shape == label_mask.shape == column_mask.shape
    assert affine.shape == (4, 4)
