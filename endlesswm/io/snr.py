"""# 📡 Signal-to-Noise Ratio Utilities

Helpers for computing per-sample SNR statistics from EndlessWM data.

The generator stores `background` and `pattern` as separate HDF5 datasets
(see `endlesswm.io.export.write_task_to_hdf5`), so any sample's SNR can
be computed without re-running the pipeline. NaN values outside the
elliptical wafer mask are ignored.

Two SNR definitions are supported:

* **energy ratio** — `E[pattern**2] / E[background**2]`
* **variance ratio** — `var(pattern) / var(background)` (insensitive to
  additive offsets in either signal)

Both are reported in linear and dB form.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_sample_snr",
    "snr_db",
    "snr_energy",
    "snr_variance",
]


def snr_energy(background: np.ndarray, pattern: np.ndarray) -> float:
    """📡 SNR as `E[pattern**2] / E[background**2]` (linear scale)."""
    p = pattern[np.isfinite(pattern)]
    b = background[np.isfinite(background)]
    if b.size == 0:
        return float("nan")
    sig = float(np.mean(p * p))
    noi = float(np.mean(b * b))
    if noi == 0.0:
        return float("inf") if sig > 0 else float("nan")
    return sig / noi


def snr_variance(background: np.ndarray, pattern: np.ndarray) -> float:
    """📡 SNR as `var(pattern) / var(background)` (linear scale)."""
    p = pattern[np.isfinite(pattern)]
    b = background[np.isfinite(background)]
    if b.size < 2:
        return float("nan")
    sig = float(np.var(p))
    noi = float(np.var(b))
    if noi == 0.0:
        return float("inf") if sig > 0 else float("nan")
    return sig / noi


def snr_db(ratio: float) -> float:
    """🔢 Convert a linear SNR to decibels (`10 * log10(ratio)`)."""
    if not np.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return float(10.0 * np.log10(ratio))


def compute_sample_snr(background: np.ndarray, pattern: np.ndarray) -> dict[str, float]:
    """📊 Convenience wrapper returning all four SNR figures for one sample."""
    e = snr_energy(background, pattern)
    v = snr_variance(background, pattern)
    return {
        "snr_energy": e,
        "snr_variance": v,
        "snr_energy_db": snr_db(e),
        "snr_variance_db": snr_db(v),
    }
