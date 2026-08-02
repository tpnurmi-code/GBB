"""Fast in-memory smoke test for the synthetic generator."""

from __future__ import annotations

import numpy as np

from gbb.synthetic import MechanisticSyntheticFMRI, SyntheticFMRIConfig


def main() -> int:
    config = SyntheticFMRIConfig.quick()
    generator = MechanisticSyntheticFMRI(config)
    result = generator.simulate_run(0, 1)

    assert result.observed_fmri.shape == (config.n_timepoints, config.num_nodes)
    assert np.all(np.isfinite(result.observed_fmri))
    assert np.any(result.stimulus > 0)
    print("GBB mechanistic synthetic smoke test passed.")
    print(f"Nodes: {config.num_nodes}")
    print(f"Edges: {generator.network.num_edges}")
    print(f"Observed shape: {result.observed_fmri.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
