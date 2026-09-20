import json

import numpy as np
import pytest

from gbb.synthetic.anatomy import build_synthetic_anatomy
from gbb.synthetic.config import SyntheticFMRIConfig
from gbb.synthetic.dynamics import build_neural_ground_truth
from gbb.synthetic.profiles import load_ground_truth_profile


def _write_profile(tmp_path, payload):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_roi_profile_overrides_node_ground_truth(tmp_path) -> None:
    config = SyntheticFMRIConfig()
    anatomy = build_synthetic_anatomy(config)

    profile_path = _write_profile(
        tmp_path,
        {
            "schema_version": 1,
            "name": "test_profile",
            "description": "Unit-test profile",
            "node_parameters": {
                "defaults": {
                    "tau_seconds": {
                        "distribution": "fixed",
                        "value": 2.5,
                    },
                    "intrinsic_drive": {
                        "distribution": "fixed",
                        "value": 0.0,
                    },
                },
                "rules": [
                    {
                        "name": "s1_override",
                        "match": {
                            "region": "S1_Postcentral",
                        },
                        "parameters": {
                            "tau_seconds": {
                                "distribution": "fixed",
                                "value": 1.75,
                            },
                            "intrinsic_drive": {
                                "distribution": "fixed",
                                "value": 0.4,
                            },
                        },
                    }
                ],
            },
        },
    )

    profile = load_ground_truth_profile(profile_path)

    truth = build_neural_ground_truth(
        config,
        anatomy,
        profile=profile,
    )

    s1 = np.asarray(anatomy.region_names) == "S1_Postcentral"
    non_s1 = ~s1

    assert np.allclose(
        truth.tau_seconds[s1],
        1.75,
    )
    assert np.allclose(
        truth.intrinsic_drive[s1],
        0.4,
    )

    assert np.allclose(
        truth.tau_seconds[non_s1],
        2.5,
    )
    assert np.allclose(
        truth.intrinsic_drive[non_s1],
        0.0,
    )


def test_profile_sampling_is_reproducible(tmp_path) -> None:
    config = SyntheticFMRIConfig()
    anatomy = build_synthetic_anatomy(config)

    profile_path = _write_profile(
        tmp_path,
        {
            "schema_version": 1,
            "name": "reproducibility",
            "node_parameters": {
                "defaults": {
                    "tau_seconds": {
                        "distribution": "normal",
                        "mean": 3.0,
                        "sd": 0.1,
                        "clip": [2.5, 3.5],
                    }
                },
                "rules": [],
            },
        },
    )

    profile = load_ground_truth_profile(profile_path)

    first = build_neural_ground_truth(
        config,
        anatomy,
        profile=profile,
    )
    second = build_neural_ground_truth(
        config,
        anatomy,
        profile=profile,
    )

    assert np.array_equal(
        first.tau_seconds,
        second.tau_seconds,
    )


def test_later_rules_override_broader_rules(tmp_path) -> None:
    config = SyntheticFMRIConfig()
    anatomy = build_synthetic_anatomy(config)

    profile_path = _write_profile(
        tmp_path,
        {
            "schema_version": 1,
            "name": "override_order",
            "node_parameters": {
                "defaults": {},
                "rules": [
                    {
                        "name": "all_s1",
                        "match": {
                            "region": "S1_Postcentral",
                        },
                        "parameters": {
                            "intrinsic_drive": {
                                "distribution": "fixed",
                                "value": 0.2,
                            }
                        },
                    },
                    {
                        "name": "left_middle_s1",
                        "match": {
                            "region": "S1_Postcentral",
                            "hemisphere": "Left",
                            "layer": "middle",
                        },
                        "parameters": {
                            "intrinsic_drive": {
                                "distribution": "fixed",
                                "value": 0.6,
                            }
                        },
                    },
                ],
            },
        },
    )

    profile = load_ground_truth_profile(profile_path)

    truth = build_neural_ground_truth(
        config,
        anatomy,
        profile=profile,
    )

    region = np.asarray(anatomy.region_names)
    hemisphere = np.asarray(anatomy.hemisphere)
    layer = np.asarray(
        [
            anatomy.layer_names[int(index)]
            for index in anatomy.layer_index
        ]
    )

    specific = (
        (region == "S1_Postcentral")
        & (hemisphere == "Left")
        & (layer == "middle")
    )

    broad_only = (
        (region == "S1_Postcentral")
        & ~specific
    )

    assert np.allclose(
        truth.intrinsic_drive[specific],
        0.6,
    )
    assert np.allclose(
        truth.intrinsic_drive[broad_only],
        0.2,
    )


def test_required_rule_matching_no_nodes_fails(tmp_path) -> None:
    config = SyntheticFMRIConfig()
    anatomy = build_synthetic_anatomy(config)

    profile_path = _write_profile(
        tmp_path,
        {
            "schema_version": 1,
            "name": "bad_roi",
            "node_parameters": {
                "defaults": {},
                "rules": [
                    {
                        "name": "misspelled_region",
                        "match": {
                            "region": "Definitely_Not_A_Region"
                        },
                        "parameters": {
                            "tau_seconds": {
                                "distribution": "fixed",
                                "value": 2.0,
                            }
                        },
                    }
                ],
            },
        },
    )

    profile = load_ground_truth_profile(profile_path)

    with pytest.raises(
        ValueError,
        match="matched no synthetic nodes",
    ):
        build_neural_ground_truth(
            config,
            anatomy,
            profile=profile,
        )


def test_optional_empty_rule_is_allowed(tmp_path) -> None:
    config = SyntheticFMRIConfig()
    anatomy = build_synthetic_anatomy(config)

    profile_path = _write_profile(
        tmp_path,
        {
            "schema_version": 1,
            "name": "optional_roi",
            "node_parameters": {
                "defaults": {},
                "rules": [
                    {
                        "name": "optional_future_region",
                        "required": False,
                        "match": {
                            "region": "Future_Region"
                        },
                        "parameters": {
                            "tau_seconds": {
                                "distribution": "fixed",
                                "value": 2.0,
                            }
                        },
                    }
                ],
            },
        },
    )

    profile = load_ground_truth_profile(profile_path)

    truth = build_neural_ground_truth(
        config,
        anatomy,
        profile=profile,
    )

    assert truth.tau_seconds.shape == (
        anatomy.num_nodes,
    )
