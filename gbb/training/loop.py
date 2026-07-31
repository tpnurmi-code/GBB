"""One-epoch training loop."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from gbb.config import config
from gbb.data.augmentation import apply_time_masking
from gbb.training.losses import (
    PearsonCorrelationLoss,
    calculate_group_lasso,
    calculate_head_sign_loss,
    calculate_hebbian_losses,
    calculate_smoothness_loss,
    calculate_temporal_orthogonality_loss,
    derivative_loss,
    tau_loss,
    transition_weighted_loss,
)
from gbb.training.metrics import (
    batch_correlation_debug,
    compute_hierarchy_index,
    get_effective_density,
)
from gbb.training.pruning import hard_prune
from gbb.training.regularizers import calculate_cfc_population_regularization
from gbb.training.schedules import get_dynamic_bioconst_lambda, neural_development


def _dataset_reference(loader):
    dataset = loader.dataset
    if isinstance(dataset, ConcatDataset):
        dataset = dataset.datasets[0]
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _log(message: str, stream) -> None:
    if stream is not None:
        print(message, file=stream, flush=True)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    writer=None,
    rank: int = 0,
    log_stream=None,
    is_master: bool = True,
    progress=None,
    scaler=None,
    use_amp: bool = False,
) -> float:
    """Train for one epoch and return average composite loss."""
    if len(loader) == 0:
        raise ValueError("Training loader is empty")

    model.train()
    ds_ref = _dataset_reference(loader)
    adjacency = ds_ref.adjacency.to(device, non_blocking=True)
    distance_matrix = getattr(ds_ref, "distance_matrix", None)
    if distance_matrix is not None:
        distance_matrix = distance_matrix.to(device, non_blocking=True)
    column_ids = ds_ref.column_ids.to(device, non_blocking=True)

    can_prune = not (
        bool(config.DEMONSTRATION) and bool(config.DISABLE_HARD_PRUNE_IN_DEMO)
    )
    minimum_prune_epoch = max(
        int(config.NUM_EPOCHS * float(config.PRUNE_START_FRAC)),
        int(config.MIN_EPOCHS_BEFORE_PRUNE),
    )
    if (
        can_prune
        and epoch >= minimum_prune_epoch
        and epoch % int(config.PRUNE_EVERY_EPOCHS) == 0
    ):
        hard_prune(model, optimizer)

    correlation_loss_fn = PearsonCorrelationLoss()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    total_loss = 0.0
    exponential_metrics: dict[str, float] = {}

    for batch_index, (fmri, stimulus, target) in enumerate(loader):
        fmri = fmri.to(device, non_blocking=True)
        stimulus = stimulus.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        model_input = fmri
        if bool(config.MASKOUT_TIMESERIES):
            mode = "global" if np.random.random() > 0.5 else "independent"
            model_input = apply_time_masking(
                fmri,
                mask_len=int(config.MASKOUT_LENGTH),
                mode=mode,
            )
        if epoch >= int(float(config.STIM_DROPOUT_START_FRAC) * config.NUM_EPOCHS):
            if torch.rand((), device=device) < float(config.STIM_DROPOUT_PROB):
                stimulus = torch.zeros_like(stimulus)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            prediction, _, hidden, _ = model(
                model_input,
                stimulus,
                adjacency,
                return_head_weights=False,
            )

        prediction = prediction.float()
        hidden = hidden.float()
        target = target.float()

        mse_anchor = criterion(prediction, target)
        if epoch >= int(config.NUM_EPOCHS * 0.75):
            mse_loss = 0.7 * mse_anchor + 0.3 * transition_weighted_loss(
                prediction, target
            )
        else:
            mse_loss = mse_anchor

        loss_correlation = correlation_loss_fn(prediction, target)
        loss_variance = F.mse_loss(
            prediction.std(dim=1, unbiased=False) + 1e-6,
            target.std(dim=1, unbiased=False) + 1e-6,
        )
        loss_derivative = derivative_loss(prediction, target)

        raw_model = model.module if hasattr(model, "module") else model
        tau_seconds = raw_model.cfc.get_tau_values()
        moderate_excess = F.relu(tau_seconds - float(config.LONGTERM_THRESHOLD_MODERATE))
        extreme_excess = F.relu(tau_seconds - float(config.LONGTERM_THRESHOLD_EXTREME))
        loss_longterm = (moderate_excess + extreme_excess.square()).mean()
        hierarchy_index = compute_hierarchy_index(adjacency, method="degree")
        tau_distribution = "lognormal" if epoch >= int(config.NUM_EPOCHS * 0.90) else "uniform"
        loss_tau, loss_rank = tau_loss(
            tau_seconds,
            dist_type=tau_distribution,
            adj=adjacency,
            hierarchy_index=hierarchy_index,
        )

        loss_metabolic = hidden.square().mean()
        loss_group = calculate_group_lasso(model, column_ids)
        loss_smoothness = prediction.new_zeros(())
        if float(config.LAMBDA_SMOOTHNESS) > 0 and distance_matrix is not None:
            loss_smoothness = calculate_smoothness_loss(
                model,
                distance_matrix,
                column_ids,
                fmri.float(),
            )

        if raw_model.kan_layers:
            coefficients = raw_model.kan_layers[0].spline_coeffs
            edge_strength = coefficients.abs().mean(dim=(2, 3))
            loss_wiring, loss_sparsity = calculate_hebbian_losses(
                model,
                edge_strength,
                raw_model.last_kan_input,
                distance_matrix,
            )
        else:
            loss_wiring = prediction.new_zeros(())
            loss_sparsity = prediction.new_zeros(())

        loss_orthogonality = calculate_temporal_orthogonality_loss(
            prediction,
            num_nodes=prediction.shape[-1],
        )
        loss_head_sign = (
            calculate_head_sign_loss(model)
            if float(config.LAMBDA_HEAD_SIGN) > 0
            and epoch >= int(float(config.HEAD_SIGN_WARMUP_FRAC) * config.NUM_EPOCHS)
            else prediction.new_zeros(())
        )
        loss_cfc_population = calculate_cfc_population_regularization(
            model,
            adj=adjacency,
        )

        sparsity_multiplier, wiring_multiplier = neural_development(
            epoch, config.NUM_EPOCHS
        )
        lambda_wiring = float(config.LAMBDA_WIRING) * wiring_multiplier
        lambda_sparsity = float(config.LAMBDA_SPARSITY) * sparsity_multiplier
        lambda_metabolic = get_dynamic_bioconst_lambda(
            epoch, config.LAMBDA_METABOLIC
        )
        lambda_group = get_dynamic_bioconst_lambda(
            epoch, config.LAMBDA_GROUP_LASSO
        )
        lambda_orthogonality = get_dynamic_bioconst_lambda(
            epoch, config.LAMBDA_ORTH_LOSS
        )
        lambda_smoothness = get_dynamic_bioconst_lambda(
            epoch, config.LAMBDA_SMOOTHNESS
        )

        regularization_start = int(float(config.REG_PHASE_START_FRAC) * config.NUM_EPOCHS)
        if epoch < regularization_start:
            lambda_correlation = float(config.LAMBDA_CORRELATION)
        else:
            phase_length = max(1, config.NUM_EPOCHS - regularization_start)
            fraction = min(1.0, (epoch - regularization_start) / phase_length)
            lambda_correlation = float(config.LAMBDA_CORRELATION) + fraction * (
                float(config.LAMBDA_CORRELATION_FINAL)
                - float(config.LAMBDA_CORRELATION)
            )

        loss = (
            float(config.LAMBDA_ACCURACY) * mse_loss
            + lambda_correlation * loss_correlation
            + float(config.LAMBDA_VAR) * loss_variance
            + float(config.LAMBDA_DERIVATIVE) * loss_derivative
            + lambda_metabolic * loss_metabolic
            + lambda_sparsity * loss_sparsity
            + lambda_group * loss_group
            + lambda_orthogonality * loss_orthogonality
            + lambda_smoothness * loss_smoothness
            + lambda_wiring * loss_wiring
            + float(config.LAMBDA_LONGTERM) * loss_longterm
            + float(config.LAMBDA_TAU_DIVERSITY) * loss_tau
            + float(config.LAMBDA_HEAD_SIGN) * loss_head_sign
            + float(config.LAMBDA_RANK_LOSS) * loss_rank
            + float(config.LAMBDA_CFC_POP) * loss_cfc_population
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.GRAD_CLIP))
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())

        correlation, _ = batch_correlation_debug(prediction, target)
        values = {
            "loss": float(loss.item()),
            "mse": float((config.LAMBDA_ACCURACY * mse_loss).item()),
            "corr": float(correlation.item()),
            "var": float((config.LAMBDA_VAR * loss_variance).item()),
            "derivative": float((config.LAMBDA_DERIVATIVE * loss_derivative).item()),
            "metabolic": float((lambda_metabolic * loss_metabolic).item()),
            "wiring": float((lambda_wiring * loss_wiring).item()),
            "sparsity": float((lambda_sparsity * loss_sparsity).item()),
            "smoothness": float((lambda_smoothness * loss_smoothness).item()),
            "group": float((lambda_group * loss_group).item()),
            "longterm": float((config.LAMBDA_LONGTERM * loss_longterm).item()),
            "tau": float((config.LAMBDA_TAU_DIVERSITY * loss_tau).item()),
            "orth": float((lambda_orthogonality * loss_orthogonality).item()),
            "mean_activity": float(prediction.mean().item()),
            "max_activity": float(prediction.max().item()),
            "pred_std": float(prediction.std(unbiased=False).item()),
        }
        for key, value in values.items():
            previous = exponential_metrics.get(key)
            exponential_metrics[key] = value if previous is None else 0.9 * previous + 0.1 * value

        if rank == 0 and writer is not None:
            global_step = epoch * len(loader) + batch_index
            writer.add_scalar("Loss/total", values["loss"], global_step)
            writer.add_scalar("Loss/mse", values["mse"], global_step)
            writer.add_scalar("Metrics/correlation", values["corr"], global_step)

        if is_master and batch_index % 10 == 0:
            message = (
                f"Mean Output Activity: {values['mean_activity']:.6g} |"
                f"Max Output Activity: {values['max_activity']:.6g}| "
                f"Corr: {values['corr']:.6g} | PredStd: {values['pred_std']:.6g}| "
                f"MSE: {values['mse']:.6g}| Corr_loss: {(lambda_correlation * loss_correlation).item():.6g}| "
                f"Var_loss: {values['var']:.6g}| Derivate loss: {values['derivative']:.6g} |"
                f"Metabolic: {values['metabolic']:.6g} | Wiring loss: {values['wiring']:.6g} |"
                f"Sparseness: {values['sparsity']:.6g} | Smoothness: {values['smoothness']:.6g}|"
                f"l_group: {values['group']:.6g} | Long term memory (tau) loss: {values['longterm']:.6g} |"
                f"tau diversity loss: {values['tau']:.6g} | temporal orthogonality loss: {values['orth']:.6g}"
            )
            _log(message, log_stream)

        if progress is not None and is_master:
            progress.set_postfix_str(
                " | ".join(
                    [
                        f"L:{exponential_metrics['loss']:.3f}",
                        f"MSE:{exponential_metrics['mse']:.3f}",
                        f"Corr:{exponential_metrics['corr']:.3f}",
                    ]
                )
            )

    density_high = get_effective_density(model, threshold=0.01)
    density_low = get_effective_density(model, threshold=0.001)
    _log(
        f"Density (>0.01): {density_high:.2f}% | Density (>0.001): {density_low:.2f}%",
        log_stream,
    )
    return total_loss / len(loader)