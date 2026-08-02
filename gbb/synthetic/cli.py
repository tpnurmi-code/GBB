"""Command-line interface for the GBB mechanistic synthetic generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SyntheticFMRIConfig
from .generator import MechanisticSyntheticFMRI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a privacy-safe GBB-compatible fMRI dataset with known "
            "mechanistic ground truth."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("synthetic_gbb_data"))
    parser.add_argument("--subjects", type=int, default=4)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--timepoints", type=int, default=160)
    parser.add_argument("--tr", type=float, default=2.5)
    parser.add_argument("--neural-dt", type=float, default=0.10)
    parser.add_argument("--response", choices=["bold", "cbv"], default="bold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate a small 2-subject, 1-run smoke-test dataset.",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Write .nii rather than .nii.gz files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        config = SyntheticFMRIConfig.quick(args.output)
        config.seed = args.seed
        config.response_kind = args.response
        config.overwrite = args.overwrite
        config.compression = not args.uncompressed
    else:
        config = SyntheticFMRIConfig(
            output_dir=args.output,
            seed=args.seed,
            overwrite=args.overwrite,
            num_subjects=args.subjects,
            num_runs=args.runs,
            num_columns=args.columns,
            n_timepoints=args.timepoints,
            tr=args.tr,
            neural_dt=args.neural_dt,
            response_kind=args.response,
            compression=not args.uncompressed,
        )

    result = MechanisticSyntheticFMRI(config).generate_dataset()
    print("Synthetic GBB dataset generated successfully.")
    print(f"Output directory: {result.output_dir}")
    print(f"Nodes: {result.anatomy.num_nodes}")
    print(f"Edges: {result.network.num_edges}")
    print(f"Runs written: {len(result.run_files)}")
    print(f"Response type: {config.response_kind.upper()}")
    print("Contains participant data: no")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
