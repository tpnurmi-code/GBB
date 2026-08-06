"""Command-line training entry point for the Glass-Box Brain model.

Run from the repository root with::

    python -m gbb.training.train

The module deliberately performs no data loading or model construction at import
 time, which keeps it import-safe for tests and Optuna workers.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from gbb.analysis.log_parser import visualize_training_dynamics
from gbb.analysis.map_export import export_map_stability
from gbb.analysis.model_outputs import visualize_results
from gbb.config import config
from gbb.data.dataset import NiftiLaminarDataset
from gbb.data.files import get_subject_files
from gbb.models.factory import build_model
from gbb.training.checkpointing import save_checkpoint
from gbb.training.distributed import (
    cleanup_distributed,
    set_seed,
    setup_distributed,
)
from gbb.training.loop import train_one_epoch
from gbb.training.validation import validate
from gbb.visualization.timeseries import visualize_prediction_dynamics


def _split_runs_by_subject(
    all_runs: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete subjects rather than individual runs.

    For a one-subject development dataset, the same runs are returned for both
    train and validation so that the code remains runnable. That fallback is not
    an independent validation design and is clearly reported to the caller.
    """
    if not all_runs:
        raise FileNotFoundError(
            f"No fMRI runs were found under {config.DATA_DIR!r}. "
            "Set GBB_DATA_DIR or edit gbb/config/defaults.py."
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("TRAIN_SET_SIZE must lie strictly between 0 and 1")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in all_runs:
        subject_id = str(run.get("subject_id") or run["id"].split("_run")[0])
        groups[subject_id].append(run)

    subject_ids = sorted(groups)
    random.Random(seed).shuffle(subject_ids)
    if len(subject_ids) == 1:
        only_runs = list(groups[subject_ids[0]])
        print(
            "WARNING: only one subject was found. The same runs will be used "
            "for training and validation; validation metrics are not independent."
        )
        return only_runs, only_runs

    split_index = round(len(subject_ids) * train_fraction)
    split_index = min(max(1, split_index), len(subject_ids) - 1)
    train_subjects = set(subject_ids[:split_index])
    test_subjects = set(subject_ids[split_index:])

    train_runs = [run for sid in train_subjects for run in groups[sid]]
    test_runs = [run for sid in test_subjects for run in groups[sid]]
    if not train_runs or not test_runs:
        raise RuntimeError("Subject-level split produced an empty train or test set")
    return train_runs, test_runs


def _loader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler=None,
    drop_last: bool,
) -> dict[str, Any]:
    """Return DataLoader arguments that are valid when ``num_workers == 0``."""
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle) if sampler is None else False,
        "sampler": sampler,
        "drop_last": bool(drop_last),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return kwargs


def _build_datasets_and_loaders(
    rank: int,
    world_size: int,
):
    all_runs = get_subject_files(config.DATA_DIR, num_runs=2)
    train_runs, test_runs = _split_runs_by_subject(
        all_runs,
        train_fraction=float(config.TRAIN_SET_SIZE),
        seed=int(config.SEED),
    )
    if rank == 0:
        print(f"Training on {len(train_runs)} runs; validating on {len(test_runs)} runs.")

    train_dataset = NiftiLaminarDataset(
        data_list=train_runs,
        mask_img=config.MASK_FILE,
        window_size=int(config.WINDOW_SIZE),
        run_type="train",
        sensory_regions=config.SENSORY_REGIONS,
    )
    test_dataset = NiftiLaminarDataset(
        data_list=test_runs,
        mask_img=config.MASK_FILE,
        window_size=int(config.WINDOW_SIZE),
        run_type="test",
        sensory_regions=config.SENSORY_REGIONS,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config.SEED),
        )
        if world_size > 1
        else None
    )
    # Every rank validates on the full set. This duplicates validation work under
    # DDP, but keeps rank-zero metrics correct without a separate all-reduce of
    # losses, correlations, and nodewise contributions.
    test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(
            batch_size=int(config.BATCH_SIZE),
            num_workers=int(config.TRAIN_LOADER_WORKERS),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=len(train_dataset) >= int(config.BATCH_SIZE),
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        **_loader_kwargs(
            batch_size=int(config.BATCH_SIZE),
            num_workers=int(config.TEST_LOADER_WORKERS),
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        ),
    )
    return all_runs, train_dataset, test_dataset, train_loader, test_loader, train_sampler


def _build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Use weight decay on connection weights, but not biases or normalization."""
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = (
            parameter.ndim < 2
            or name.endswith("bias")
            or "norm" in name.lower()
            or "node_bias" in name
        )
        (no_decay_parameters if no_decay else decay_parameters).append(parameter)

    parameter_groups = []
    if decay_parameters:
        parameter_groups.append(
            {
                "params": decay_parameters,
                "weight_decay": float(config.WEIGHT_DECAY),
            }
        )
    if no_decay_parameters:
        parameter_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})
    return torch.optim.AdamW(parameter_groups, lr=float(config.LEARNING_RATE))


def _make_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def main() -> None:
    rank, world_size, local_rank = setup_distributed()
    is_master = rank == 0
    set_seed(int(config.SEED) + rank)
    device = _make_device(local_rank)

    results_dir = Path(config.RESULTS_DIR)
    log_dir = Path(config.LOG_DIR)
    if is_master:
        results_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    log_stream = None
    try:
        (
            all_runs,
            train_dataset,
            test_dataset,
            train_loader,
            test_loader,
            train_sampler,
        ) = _build_datasets_and_loaders(rank, world_size)

        model = build_model(
            num_nodes=train_dataset.num_nodes,
            time_points=int(config.WINDOW_SIZE),
            sensory_mask=train_dataset.sensory_mask,
            model_type=config.MODEL_TYPE,
            use_hemodynamic_head=bool(config.USE_HEMODYNAMIC_HEAD),
            allow_all_nodes=bool(config.ALLOW_ALL_NODES_STIMULUS),
        ).to(device)

        if world_size > 1:
            ddp_kwargs = {"find_unused_parameters": False}
            if device.type == "cuda":
                ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
            model = DDP(model, **ddp_kwargs)

        optimizer = _build_optimizer(model)
        criterion = nn.MSELoss()
        use_amp = bool(config.USE_AMP) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"training_{timestamp}.txt"
        if is_master:
            log_stream = log_path.open("w", encoding="utf-8", buffering=1)
            writer = SummaryWriter(log_dir=str(log_dir / timestamp))
            print(f"Starting {config.MODEL_TYPE} training on {device}; log={log_path}")

        best_correlation = float("-inf")
        epochs_without_improvement = 0
        final_node_values = None
        started = time.time()
        progress = tqdm(
            range(int(config.NUM_EPOCHS)),
            desc="Epochs",
            disable=not is_master,
        )

        for epoch in progress:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                writer=writer,
                rank=rank,
                log_stream=log_stream,
                is_master=is_master,
                progress=progress,
                scaler=scaler,
                use_amp=use_amp,
            )

            should_validate = (
                epoch % max(1, int(config.VALIDATE_EVERY_EPOCHS)) == 0
                or epoch == int(config.NUM_EPOCHS) - 1
            )
            if not should_validate:
                continue

            (
                test_loss,
                test_correlation,
                isolated_correlation,
                connection_contribution,
                final_node_values,
            ) = validate(
                model,
                test_loader,
                criterion,
                device,
                ds_ref=test_dataset,
                use_amp=use_amp,
            )
            if is_master:
                epoch_line = (
                    f"Epoch {epoch} | Train: {train_loss:.6g} | "
                    f"Test: {test_loss:.6g} | Corr: {test_correlation:.6g}"
                )
                print(epoch_line)
                print(epoch_line, file=log_stream, flush=True)
                if writer is not None:
                    writer.add_scalar("Validation/loss", test_loss, epoch)
                    writer.add_scalar("Validation/correlation", test_correlation, epoch)
                    writer.add_scalar(
                        "Validation/isolated_correlation",
                        isolated_correlation,
                        epoch,
                    )
                    writer.add_scalar(
                        "Validation/connection_contribution",
                        connection_contribution,
                        epoch,
                    )

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    test_loss,
                    results_dir / "checkpoint_latest.pth",
                )
                if test_correlation > best_correlation:
                    best_correlation = test_correlation
                    epochs_without_improvement = 0
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        test_loss,
                        results_dir / "checkpoint_best.pth",
                    )
                else:
                    epochs_without_improvement += max(1, int(config.VALIDATE_EVERY_EPOCHS))

            # All ranks must take the same early-stopping branch.
            stop_tensor = torch.tensor(
                int(epochs_without_improvement >= int(config.PATIENCE)),
                device=device,
            )
            if world_size > 1:
                torch.distributed.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                if is_master:
                    print(f"Early stopping at epoch {epoch}.")
                break

        final_loss, final_corr, final_iso, final_contribution, final_node_values = validate(
            model,
            test_loader,
            criterion,
            device,
            ds_ref=test_dataset,
            use_amp=use_amp,
        )
        if is_master:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                final_loss,
                results_dir / "checkpoint_final.pth",
            )
            elapsed = time.time() - started
            print(
                f"Training finished in {elapsed:.1f}s | loss={final_loss:.6g} | "
                f"corr={final_corr:.6g} | isolated_corr={final_iso:.6g}"
            )
            date_string = datetime.now().strftime("%d_%m_%Y_%H_%M")

            # Export is intentionally non-fatal: successful training and a saved
            # checkpoint must not be discarded because an optional plotting
            # dependency or a display backend is unavailable.
            try:
                visualize_results(
                    model,
                    train_dataset,
                    world_size,
                    all_runs,
                    final_node_values,
                    test_loader,
                    date_string,
                )
                visualize_prediction_dynamics(
                    model,
                    test_dataset,
                    device,
                    results_dir / f"prediction_dynamics_{date_string}.png",
                )
                visualize_training_dynamics(
                    log_path,
                    results_dir / f"training_dynamics_{date_string}.png",
                    int(config.NUM_EPOCHS),
                    title="GBB training dynamics",
                )
                export_map_stability(
                    model,
                    test_loader,
                    results_dir,
                    datestring=date_string,
                )
            except Exception as exc:  # noqa: BLE001 - export should be non-fatal
                print(f"WARNING: optional post-training export failed: {exc}")
    finally:
        if log_stream is not None:
            log_stream.close()
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
