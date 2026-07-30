import optuna
import torch
import torch.nn as nn
import numpy as np
import config
from models import MesocortGBB
from dataset import NiftiLaminarDataset
from train import train_one_epoch, validate
from utils import get_subject_files
from torch.utils.data import DataLoader


def objective(trial):
    # Full Sweep of Models H1-H6
    config.MODEL_TYPE = trial.suggest_categorical("model_type", ["H1", "H2", "H3", "H4", "H5", "H6"])
    config.FEAT_EXT_HIDDEN = trial.suggest_categorical("feat_ext_hidden", [32, 64])
    config.KAN_BASIS_FUNCTIONS = trial.suggest_int("kan_basis", 3, 6)
    config.KAN_LAYERS = trial.suggest_int("kan_layers", 1, 2)
    config.KAN_HEADS = trial.suggest_int("kan_heads", 1, 4)
    config.CFC_BACKBONE_UNITS = trial.suggest_categorical("cfc_units", [64, 128])

    config.LAMBDA_METABOLIC = trial.suggest_float("lambda_meta", 1e-5, 1e-2, log=True)
    config.LAMBDA_SPARSITY = trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True)
    config.LAMBDA_HEAD_GROUP = trial.suggest_float("lambda_group", 1e-4, 1e-2, log=True)
    config.LAMBDA_SMOOTHNESS = trial.suggest_float("lambda_smooth", 1e-5, 1e-2, log=True)
    config.LAMBDA_WIRING = trial.suggest_float("lambda_wire", 1e-5, 1e-2, log=True)

    device = config.DEVICE
    all_runs = get_subject_files(config.DATA_DIR, num_runs=1)
    if not all_runs: return -100.0
    
    run = all_runs[0]
    ds = NiftiLaminarDataset(run['fmri'], run['mask'], run['events'], config.TR, config.WINDOW_SIZE)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    
    try: model = MesocortGBB(ds.num_nodes, config.WINDOW_SIZE).to(device)
    except: return -100.0 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    final_mse = 1.0
    for epoch in range(config.TRIALS_PER_EVAL):
        loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch, None, 0)
        final_mse = validate(model, loader, criterion, device)
        trial.report(final_mse, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    perf_drop = (final_mse - config.BASELINE_MSE) / config.BASELINE_MSE
    if perf_drop > config.MAX_PERF_DROP: return -100.0 
        
    bio_score = (np.log(config.LAMBDA_METABOLIC) + np.log(config.LAMBDA_SPARSITY) + np.log(config.LAMBDA_HEAD_GROUP))
    return bio_score

if __name__ == "__main__":
    print(f"🚀 Starting Real Optimization loop on {config.DEVICE}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    print("🏆 Best Trial:", study.best_params)