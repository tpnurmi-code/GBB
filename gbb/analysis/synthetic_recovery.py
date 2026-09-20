"""Evaluate recovery of synthetic node-level mechanistic parameters."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


@dataclass(frozen=True, slots=True)
class RecoveryTarget:
    """Mapping between one synthetic parameter and one learned GBB map."""

    key: str
    display_name: str
    ground_truth_npz_key: str
    ground_truth_map_stem: str
    learned_filename: str
    unit: str | None = None


DEFAULT_NODE_RECOVERY_TARGETS: tuple[RecoveryTarget, ...] = (
    RecoveryTarget(
        key="tau",
        display_name="CfC effective timescale",
        ground_truth_npz_key="tau_seconds",
        ground_truth_map_stem="GT_CfC_Tau_s",
        learned_filename="Map_CfC_Tau_s.npy",
        unit="s",
    ),
    RecoveryTarget(
        key="intrinsic_drive",
        display_name="CfC intrinsic drive",
        ground_truth_npz_key="intrinsic_drive",
        ground_truth_map_stem="GT_CfC_IntrinsicDrive",
        learned_filename="Map_CfC_IntrinsicDrive.npy",
        unit=None,
    ),
)


TARGETS_BY_KEY = {
    target.key: target
    for target in DEFAULT_NODE_RECOVERY_TARGETS
}


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Scalar recovery statistics for one node-level parameter."""

    n_nodes: int

    pearson_r: float
    spearman_rho: float

    mae: float
    rmse: float

    regression_slope: float
    regression_intercept: float

    lin_ccc: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _as_finite_vector(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if vector.size == 0:
        raise ValueError(
            f"{name} contains no values"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return vector


def _is_constant(values: np.ndarray) -> bool:
    if values.size < 2:
        return True

    return bool(
        np.allclose(
            values,
            values[0],
        )
    )


def _lin_concordance_correlation(
    truth: np.ndarray,
    learned: np.ndarray,
) -> float:
    """Lin's concordance correlation coefficient."""

    truth_mean = float(np.mean(truth))
    learned_mean = float(np.mean(learned))

    truth_centered = truth - truth_mean
    learned_centered = learned - learned_mean

    truth_variance = float(
        np.mean(truth_centered**2)
    )
    learned_variance = float(
        np.mean(learned_centered**2)
    )

    covariance = float(
        np.mean(
            truth_centered
            * learned_centered
        )
    )

    denominator = (
        truth_variance
        + learned_variance
        + (truth_mean - learned_mean) ** 2
    )

    if np.isclose(denominator, 0.0):
        if np.allclose(truth, learned):
            return 1.0

        return float("nan")

    return float(
        2.0 * covariance
        / denominator
    )


def compute_recovery_metrics(
    truth: np.ndarray,
    learned: np.ndarray,
) -> RecoveryMetrics:
    """Compute node-wise parameter-recovery statistics."""

    truth_vector = _as_finite_vector(
        truth,
        name="ground truth",
    )

    learned_vector = _as_finite_vector(
        learned,
        name="learned map",
    )

    if truth_vector.shape != learned_vector.shape:
        raise ValueError(
            "Ground-truth and learned vectors must have identical shapes: "
            f"{truth_vector.shape} != {learned_vector.shape}"
        )

    n_nodes = int(truth_vector.size)

    truth_constant = _is_constant(
        truth_vector
    )
    learned_constant = _is_constant(
        learned_vector
    )

    if (
        n_nodes >= 2
        and not truth_constant
        and not learned_constant
    ):
        pearson_r = float(
            pearsonr(
                truth_vector,
                learned_vector,
            ).statistic
        )

        spearman_rho = float(
            spearmanr(
                truth_vector,
                learned_vector,
            ).statistic
        )
    else:
        pearson_r = float("nan")
        spearman_rho = float("nan")

    error = learned_vector - truth_vector

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )

    if (
        n_nodes >= 2
        and not truth_constant
    ):
        regression = linregress(
            truth_vector,
            learned_vector,
        )

        regression_slope = float(
            regression.slope
        )

        regression_intercept = float(
            regression.intercept
        )

    else:
        regression_slope = float("nan")
        regression_intercept = float("nan")

    lin_ccc = _lin_concordance_correlation(
        truth_vector,
        learned_vector,
    )

    return RecoveryMetrics(
        n_nodes=n_nodes,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        mae=mae,
        rmse=rmse,
        regression_slope=regression_slope,
        regression_intercept=regression_intercept,
        lin_ccc=lin_ccc,
    )


def _resolve_ground_truth_archive(
    path: str | Path,
) -> tuple[Path, Path]:
    """Resolve dataset root, ground_truth directory, or NPZ archive."""

    supplied = Path(path).expanduser().resolve()

    if supplied.is_file():
        if supplied.suffix != ".npz":
            raise ValueError(
                "A ground-truth file must be an NPZ archive"
            )

        return supplied, supplied.parent

    direct = (
        supplied
        / "mechanistic_ground_truth.npz"
    )

    nested = (
        supplied
        / "ground_truth"
        / "mechanistic_ground_truth.npz"
    )

    if direct.is_file():
        return direct, supplied

    if nested.is_file():
        return nested, supplied / "ground_truth"

    raise FileNotFoundError(
        "Could not find mechanistic_ground_truth.npz under "
        f"{supplied}"
    )


def _load_node_metadata(
    ground_truth_directory: Path,
    *,
    n_nodes: int,
) -> pd.DataFrame:
    metadata_path = (
        ground_truth_directory
        / "node_ground_truth.csv"
    )

    if not metadata_path.is_file():
        return pd.DataFrame(
            {
                "node_id": np.arange(
                    1,
                    n_nodes + 1,
                    dtype=np.int64,
                )
            }
        )

    metadata = pd.read_csv(
        metadata_path
    )

    if len(metadata) != n_nodes:
        raise ValueError(
            "node_ground_truth.csv contains "
            f"{len(metadata)} rows but recovery vectors contain "
            f"{n_nodes} nodes"
        )

    preferred_columns = [
        "node_id",
        "label",
        "hemisphere",
        "network_id",
        "column_id",
        "layer",
        "hierarchy",
        "spatial_gradient",
    ]

    available = [
        column
        for column in preferred_columns
        if column in metadata.columns
    ]

    return metadata.loc[
        :,
        available,
    ].copy()


def _json_safe_metrics(
    metrics: RecoveryMetrics,
) -> dict[str, int | float | None]:
    payload: dict[str, int | float | None] = {}

    for key, value in metrics.to_dict().items():
        if isinstance(value, float) and not np.isfinite(value):
            payload[key] = None
        else:
            payload[key] = value

    return payload


def evaluate_synthetic_recovery(
    ground_truth: str | Path,
    learned_directory: str | Path,
    output_directory: str | Path,
    *,
    targets: Sequence[RecoveryTarget] = DEFAULT_NODE_RECOVERY_TARGETS,
) -> pd.DataFrame:
    """Compare learned GBB node maps with known synthetic ground truth."""

    archive_path, ground_truth_directory = (
        _resolve_ground_truth_archive(
            ground_truth
        )
    )

    learned_directory = (
        Path(learned_directory)
        .expanduser()
        .resolve()
    )

    if not learned_directory.is_dir():
        raise FileNotFoundError(
            f"Learned-map directory does not exist: "
            f"{learned_directory}"
        )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    report_targets: list[
        dict[str, object]
    ] = []

    with np.load(
        archive_path,
        allow_pickle=False,
    ) as ground_truth_archive:
        for target in targets:
            if (
                target.ground_truth_npz_key
                not in ground_truth_archive
            ):
                raise KeyError(
                    f"Ground-truth archive does not contain "
                    f"{target.ground_truth_npz_key!r}"
                )

            learned_path = (
                learned_directory
                / target.learned_filename
            )

            if not learned_path.is_file():
                raise FileNotFoundError(
                    f"Learned map does not exist: "
                    f"{learned_path}"
                )

            truth_vector = _as_finite_vector(
                ground_truth_archive[
                    target.ground_truth_npz_key
                ],
                name=(
                    f"ground truth "
                    f"{target.ground_truth_npz_key}"
                ),
            )

            learned_vector = _as_finite_vector(
                np.load(
                    learned_path,
                    allow_pickle=False,
                ),
                name=(
                    f"learned map "
                    f"{target.learned_filename}"
                ),
            )

            metrics = compute_recovery_metrics(
                truth_vector,
                learned_vector,
            )

            metadata = _load_node_metadata(
                ground_truth_directory,
                n_nodes=metrics.n_nodes,
            )

            node_table = metadata.copy()

            node_table["ground_truth"] = (
                truth_vector
            )

            node_table["learned"] = (
                learned_vector
            )

            node_table["error"] = (
                learned_vector
                - truth_vector
            )

            node_table["absolute_error"] = (
                np.abs(
                    learned_vector
                    - truth_vector
                )
            )

            node_table.to_csv(
                output_directory
                / f"recovery_{target.key}_nodes.csv",
                index=False,
            )

            metrics_payload = metrics.to_dict()

            summary_rows.append(
                {
                    "target": target.key,
                    "display_name": (
                        target.display_name
                    ),
                    "unit": (
                        target.unit
                        if target.unit is not None
                        else ""
                    ),
                    **metrics_payload,
                }
            )

            report_targets.append(
                {
                    "target": target.key,
                    "display_name": (
                        target.display_name
                    ),
                    "unit": target.unit,
                    "ground_truth_npz_key": (
                        target.ground_truth_npz_key
                    ),
                    "ground_truth_map_stem": (
                        target.ground_truth_map_stem
                    ),
                    "learned_filename": (
                        target.learned_filename
                    ),
                    "metrics": (
                        _json_safe_metrics(
                            metrics
                        )
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_directory
        / "recovery_summary.csv",
        index=False,
    )

    report = {
        "schema_version": 1,
        "ground_truth_archive": (
            str(archive_path)
        ),
        "learned_directory": (
            str(learned_directory)
        ),
        "output_directory": (
            str(output_directory)
        ),
        "ground_truth_level": (
            "base_population"
        ),
        "node_ordering": (
            "Synthetic node index 0 corresponds to atlas label 1. "
            "Learned maps must originate from the same labelled mask."
        ),
        "targets": report_targets,
    }

    (
        output_directory
        / "recovery_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned GBB parameter maps with "
            "known synthetic ground truth."
        )
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help=(
            "Synthetic dataset root, ground_truth directory, "
            "or mechanistic_ground_truth.npz."
        ),
    )

    parser.add_argument(
        "--learned",
        required=True,
        type=Path,
        help=(
            "Directory containing learned Map_*.npy files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<learned>/synthetic_recovery."
        ),
    )

    parser.add_argument(
        "--target",
        dest="target_keys",
        action="append",
        choices=sorted(
            TARGETS_BY_KEY
        ),
        default=None,
        help=(
            "Evaluate only the selected target. "
            "May be supplied multiple times."
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(
        argv
    )

    learned_directory = (
        Path(args.learned)
        .expanduser()
        .resolve()
    )

    output_directory = (
        Path(args.output)
        if args.output is not None
        else learned_directory
        / "synthetic_recovery"
    )

    if args.target_keys is None:
        targets = (
            DEFAULT_NODE_RECOVERY_TARGETS
        )
    else:
        targets = tuple(
            TARGETS_BY_KEY[key]
            for key in args.target_keys
        )

    summary = evaluate_synthetic_recovery(
        args.ground_truth,
        learned_directory,
        output_directory,
        targets=targets,
    )

    print(
        "Synthetic parameter-recovery "
        "evaluation complete."
    )
    print(
        f"Output directory: "
        f"{Path(output_directory).resolve()}"
    )
    print()
    print(
        summary.to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())