"""Optuna search using the refactored GBB APIs.

This script intentionally uses one process per trial. Mutating the flat runtime
configuration inside a trial is therefore safe for sequential optimization; do
not set ``n_jobs`` above one without moving trial values into explicit config
objects passed to model constructors and training functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from gbb.config import config
from gbb.data.dataset import NiftiLaminarDataset
from gbb.data.files import get_subject_files
from gbb.models.factory import build_model
from gbb.training.loop import train_one_epoch
from gbb.training.train import _split_runs_by_subject
from gbb.training.validation import validate


def _trial_loaders():
    runs = get_subject_files(config.DATA_DIR, num_runs=1)
    train_runs, validation_runs = _split_runs_by_subject(
        runs,
        train_fraction=float(config.TRAIN_SET_SIZE),
        seed=int(config.SEED),
    )
    train_dataset = NiftiLaminarDataset(
        train_runs,
        config.MASK_FILE,
        window_size=int(config.WINDOW_SIZE),
        run_type="train",
        sensory_regions=config.SENSORY_REGIONS,
    )
    validation_dataset = NiftiLaminarDataset(
        validation_runs,
        config.MASK_FILE,
        window_size=int(config.WINDOW_SIZE),
        run_type="test",
        sensory_regions=config.SENSORY_REGIONS,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=True,
        num_workers=0,
        drop_last=len(train_dataset) >= int(config.BATCH_SIZE),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=False,
        num_workers=0,
    )
    return train_dataset, validation_dataset, train_loader, validation_loader


def objective(trial: optuna.Trial) -> float:
    config.MODEL_TYPE = trial.suggest_categorical(
        "model_type", ["H1", "H2", "H3", "H4", "H5", "H6"]
    )
    config.FEAT_EXT_HIDDEN = trial.suggest_categorical(
        "feature_hidden", [32, 64, 128]
    )
    config.KAN_BASIS_FUNCTIONS = trial.suggest_int("kan_basis", 3, 8)
    config.KAN_LAYERS = trial.suggest_int("kan_layers", 1, 2)
    config.KAN_HEADS = trial.suggest_int("kan_heads", 1, 5)
    config.CFC_BACKBONE_UNITS = trial.suggest_categorical(
        "cfc_units", [64, 128]
    )
    config.LAMBDA_METABOLIC = trial.suggest_float(
        "lambda_metabolic", 1e-6, 1e-2, log=True
    )
    config.LAMBDA_SPARSITY = trial.suggest_float(
        "lambda_sparsity", 1e-7, 1e-3, log=True
    )
    config.LAMBDA_HEAD_GROUP = trial.suggest_float(
        "lambda_head_group", 1e-6, 1e-2, log=True
    )
    config.LAMBDA_SMOOTHNESS = trial.suggest_float(
        "lambda_smoothness", 1e-5, 10.0, log=True
    )
    config.LAMBDA_WIRING = trial.suggest_float(
        "lambda_wiring", 1e-7, 1e-2, log=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, validation_dataset, train_loader, validation_loader = _trial_loaders()
    model = build_model(
        train_dataset.num_nodes,
        int(config.WINDOW_SIZE),
        train_dataset.sensory_mask,
        model_type=config.MODEL_TYPE,
        hidden_dim=int(config.FEAT_EXT_HIDDEN),
        use_hemodynamic_head=bool(config.USE_HEMODYNAMIC_HEAD),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LEARNING_RATE),
        weight_decay=float(config.WEIGHT_DECAY),
    )
    criterion = nn.MSELoss()
    use_amp = bool(config.USE_AMP) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_correlation = float("-inf")
    for epoch in range(int(config.TRIALS_PER_EVAL)):
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            is_master=False,
            scaler=scaler,
            use_amp=use_amp,
        )
        _, correlation, _, _, _ = validate(
            model,
            validation_loader,
            criterion,
            device,
            ds_ref=validation_dataset,
            use_amp=use_amp,
        )
        best_correlation = max(best_correlation, correlation)
        trial.report(correlation, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_correlation


def main() -> None:
    sampler = optuna.samplers.TPESampler(seed=int(config.SEED))
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="gbb_phase1",
    )
    study.optimize(objective, n_trials=int(config.N_TRIALS), n_jobs=1)

    output = Path(config.RESULTS_DIR) / "optimization"
    output.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    (output / "best_trial.json").write_text(
        json.dumps(
            {
                "value": study.best_value,
                "parameters": study.best_params,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Best validation correlation:", study.best_value)
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()