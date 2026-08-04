"""Parse text training logs and render a multi-panel summary."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

BATCH_RE = re.compile(
    r"Mean Output Activity:\s*(?P<mean_act>[-+0-9.eE]+)\s*\|"
    r"Max Output Activity:\s*(?P<max_act>[-+0-9.eE]+)\|"
    r"\s*Corr:\s*(?P<corr>[-+0-9.eE]+)\s*\|"
    r"\s*PredStd:\s*(?P<pred_std>[-+0-9.eE]+)\|"
    r"\s*MSE:\s*(?P<mse>[-+0-9.eE]+)\|"
    r"\s*Corr_loss:\s*(?P<corr_loss>[-+0-9.eE]+)\|"
    r"\s*Var_loss:\s*(?P<var_loss>[-+0-9.eE]+)\|"
    r"\s*Derivate loss:\s*(?P<derivative_loss>[-+0-9.eE]+)\s*\|"
    r"\s*Metabolic:\s*(?P<metabolic>[-+0-9.eE]+)\s*\|"
    r"\s*Wiring loss:\s*(?P<wiring_loss>[-+0-9.eE]+)\s*\|"
    r"\s*Sparseness:\s*(?P<sparseness>[-+0-9.eE]+)\s*\|"
    r"\s*Smoothness:\s*(?P<smoothness>[-+0-9.eE]+)\|"
    r"\s*l_group:\s*(?P<l_group>[-+0-9.eE]+)\s*\|"
    r"\s*Long term memory \(tau\) loss:\s*(?P<longterm_tau>[-+0-9.eE]+)\s*\|"
    r"\s*tau diversity loss:\s*(?P<tau_diversity>[-+0-9.eE]+)\s*\|"
    r"\s*temporal orthogonality loss:\s*(?P<temporal_orthogonality>[-+0-9.eE]+)"
)

DENSITY_RE = re.compile(
    r"Density \(>0\.01\):\s*(?P<density1>[-+0-9.eE]+)%\s*\|\s*Density \(>0\.001\):\s*(?P<density2>[-+0-9.eE]+)%"
)

EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*Train:\s*(?P<train>[-+0-9.eE]+)\s*\|\s*Test:\s*(?P<test>[-+0-9.eE]+)\s*\|\s*Corr:\s*(?P<val_corr>[-+0-9.eE]+)"
)


LOSS_KEYS = [
    "mse",
    "corr_loss",
    "var_loss",
    "derivative_loss",
    "metabolic",
    "wiring_loss",
    "sparseness",
    "smoothness",
    "l_group",
    "longterm_tau",
    "tau_diversity",
    "temporal_orthogonality",
]





def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")





def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s).strip()

def parse_log(log_path: Path):
    lines = [strip_ansi(line) for line in log_path.read_text(errors="ignore").splitlines()]
    lines = [line for line in lines if line]

    epochs = []
    current_batches = []
    current_epoch_idx = 0

    def finalize_epoch(density1=None, density2=None):
        nonlocal current_batches, current_epoch_idx, epochs
        if not current_batches:
            # still create empty epoch slot if density exists
            if density1 is None and density2 is None:
                return
            epochs.append({
                "epoch": current_epoch_idx,
                "mean_act": np.nan,
                "max_act": np.nan,
                "corr": np.nan,
                "pred_std": np.nan,
                "density1": density1,
                "density2": density2,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "val_corr": np.nan,
                "loss_components": {k: np.nan for k in LOSS_KEYS},
            })
            current_epoch_idx += 1
            return

        epoch_record = {
            "epoch": current_epoch_idx,
            "mean_act": mean_or_nan([b["mean_act"] for b in current_batches]),
            "max_act": mean_or_nan([b["max_act"] for b in current_batches]),
            "corr": mean_or_nan([b["corr"] for b in current_batches]),
            "pred_std": mean_or_nan([b["pred_std"] for b in current_batches]),
            "density1": density1,
            "density2": density2,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_corr": np.nan,
            "loss_components": {k: mean_or_nan([b[k] for b in current_batches]) for k in LOSS_KEYS},
        }
        epochs.append(epoch_record)
        current_batches = []
        current_epoch_idx += 1

    for line in lines:
        m = BATCH_RE.search(line)
        if m:
            batch = {k: float(v) for k, v in m.groupdict().items()}
            current_batches.append(batch)
            continue

        m = DENSITY_RE.search(line)
        if m:
            density1 = float(m.group("density1"))
            density2 = float(m.group("density2"))
            finalize_epoch(density1, density2)
            continue

        m = EPOCH_RE.search(line)
        if m:
            e = int(m.group("epoch"))
            # Attach validation summary to explicit epoch if it exists,
            # otherwise to the most recent finalized epoch.
            target_idx = e if e < len(epochs) else max(len(epochs) - 1, 0)
            if len(epochs) == 0:
                epochs.append({
                    "epoch": e,
                    "mean_act": np.nan,
                    "max_act": np.nan,
                    "corr": np.nan,
                    "pred_std": np.nan,
                    "density1": np.nan,
                    "density2": np.nan,
                    "train_loss": float(m.group("train")),
                    "val_loss": float(m.group("test")),
                    "val_corr": float(m.group("val_corr")),
                    "loss_components": {k: np.nan for k in LOSS_KEYS},
                })
            else:
                epochs[target_idx]["train_loss"] = float(m.group("train"))
                epochs[target_idx]["val_loss"] = float(m.group("test"))
                epochs[target_idx]["val_corr"] = float(m.group("val_corr"))
            continue

    # finalize any trailing partial epoch
    if current_batches:
        finalize_epoch(np.nan, np.nan)

    return epochs



def make_figure(epochs, total_epochs, output_path: Path, title: str ):
    if not epochs:
        raise RuntimeError("No epochs could be parsed from the log.")

    epoch_nums = np.array([ep["epoch"] for ep in epochs], dtype=float)
    if total_epochs <= 1:
        x_pct = epoch_nums
        x_label = "Epoch"
    else:
        x_pct = 100.0 * epoch_nums / (total_epochs - 1)
        x_label = "Training progress (% of configured epochs)"

    corr_train = np.array([ep["corr"] for ep in epochs], dtype=float)
    corr_val = np.array([ep["val_corr"] for ep in epochs], dtype=float)

    mean_act = np.array([ep["mean_act"] for ep in epochs], dtype=float)
    max_act = np.array([ep["max_act"] for ep in epochs], dtype=float)

    density1 = np.array([ep["density1"] for ep in epochs], dtype=float)
    density2 = np.array([ep["density2"] for ep in epochs], dtype=float)

    checkpoints = build_bottom_contribution_data(epochs, total_epochs)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.25], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0])

    # 1) Correlation
    ax1.plot(x_pct, corr_train, label="Train batch correlation")
    ax1.plot(x_pct, corr_val, label="Validation correlation")
    ax1.set_ylabel("Correlation")
    ax1.set_title("Correlation over training")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Activity
    ax2.plot(x_pct, mean_act, label="Mean output activity")
    ax2.plot(x_pct, max_act, label="Max output activity")
    ax2.set_ylabel("Activity")
    ax2.set_title("Output activity over training")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Densities
    ax3.plot(x_pct, density1, label="Density (>0.01)")
    ax3.plot(x_pct, density2, label="Density (>0.001)")
    ax3.set_ylabel("Density (%)")
    ax3.set_xlabel(x_label)
    ax3.set_title("Effective graph density over training")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4) Loss contribution bars
    x = np.arange(len(checkpoints))
    bottom = np.zeros(len(checkpoints), dtype=float)

    for key in LOSS_KEYS:
        vals = np.array([cp["shares"][key] * 100.0 for cp in checkpoints], dtype=float)
        ax4.bar(x, vals, bottom=bottom, label=key)
        bottom += vals

    labels = [
        f"{int(cp['fraction']*100)}%\n(target E{cp['target_epoch']}, used E{cp['actual_epoch']})"
        for cp in checkpoints
    ]
    ax4.set_xticks(x, labels)
    ax4.set_ylabel("Contribution to logged loss sum (%)")
    ax4.set_title("Relative loss-component contributions at 25%, 50%, 75%, and 100% of training")
    ax4.grid(True, axis="y", alpha=0.3)

    # Compact legend outside plot
    ax4.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.suptitle(title or f"Training dynamics summary: {output_path.stem}", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def nearest_epoch_index(epochs, target_epoch):
    epoch_nums = np.array([ep["epoch"] for ep in epochs], dtype=float)
    return int(np.argmin(np.abs(epoch_nums - target_epoch)))


def build_bottom_contribution_data(epochs, total_epochs):
    checkpoint_specs = [0.25, 0.50, 0.75, 1.00]
    selections = []

    for frac in checkpoint_specs:
        target_epoch = max(0, int(round(frac * total_epochs)) - 1)
        idx = nearest_epoch_index(epochs, target_epoch)
        ep = epochs[idx]
        contrib = {k: max(0.0, float(ep["loss_components"].get(k, np.nan))) for k in LOSS_KEYS}
        total = sum(v for v in contrib.values() if np.isfinite(v))
        if total <= 0:
            shares = {k: 0.0 for k in LOSS_KEYS}
        else:
            shares = {k: (v / total) for k, v in contrib.items()}
        selections.append({
            "fraction": frac,
            "target_epoch": target_epoch,
            "actual_epoch": ep["epoch"],
            "shares": shares,
        })

    return selections


def visualize_training_dynamics(log_path, output_path, total_epochs, title=None):
    epochs = parse_log(Path(log_path))
    make_figure(epochs, total_epochs, Path(output_path), title)