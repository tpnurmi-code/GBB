"""Distributed and reproducibility setup."""

from __future__ import annotations

import os
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _slurm_master_address() -> str:
    node_list = os.environ.get("SLURM_NODELIST")
    if node_list and shutil.which("scontrol"):
        result = subprocess.run(
            ["scontrol", "show", "hostnames", node_list],
            check=True,
            capture_output=True,
            text=True,
        )
        hostnames = result.stdout.split()
        if hostnames:
            return hostnames[0]
    return "127.0.0.1"


def setup_distributed() -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` for SLURM, torchrun, or local use."""
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        os.environ.setdefault("MASTER_ADDR", _slurm_master_address())
        os.environ.setdefault("MASTER_PORT", "29500")
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        return 0, 1, 0

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()