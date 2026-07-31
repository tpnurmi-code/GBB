"""Canonical response kernels used for optional stimulus preprocessing."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from gbb.config import config


def canonical_hrf_response(time: float = 32.0, tr: float | None = None) -> np.ndarray:
    """Return a normalized positive gamma approximation of a canonical HRF."""
    tr = float(config.TR if tr is None else tr)
    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")

    time_axis = np.arange(0.0, max(float(time), tr), tr, dtype=np.float32)
    kernel = gamma.pdf(time_axis, 7.0, scale=1.0).astype(np.float32)
    total = float(kernel.sum())
    if total <= 1e-9:
        kernel = np.zeros_like(time_axis)
        kernel[0] = 1.0
    else:
        kernel /= total
    return kernel


def canonical_cbv_response(
    tr: float,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Return a normalized gamma approximation of a canonical CBV response."""
    tr = float(tr)
    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")

    time_axis = np.arange(0.0, max(float(time_length), tr), tr, dtype=np.float32) - onset
    response = gamma.pdf(time_axis, 4.0, scale=1.0).astype(np.float32)
    response[time_axis < 0] = 0.0
    peak = float(response.max(initial=0.0))
    if peak > 0:
        response /= peak
    else:
        response[0] = 1.0
    return response