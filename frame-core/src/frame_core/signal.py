"""
Signal-processing primitives shared across all metric layers.

Conventions (fixed project-wide, see paper methods):
  * Butterworth low-pass, cutoff 6 Hz, order 4, zero-phase (filtfilt).
  * Time-normalisation ALWAYS by linear interpolation (np.interp), NEVER FFT
    resample -- FFT resample (scipy.signal.resample) rings at the endpoints of
    non-periodic trajectories (Gibbs), which corrupts peak amplitudes.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

CUTOFF_HZ = 6.0
FS_HZ = 240.0
ORDER = 4


def butter_lowpass(x, cutoff=CUTOFF_HZ, fs=FS_HZ, order=ORDER):
    """Zero-phase Butterworth low-pass along axis 0. Short signals are passed through."""
    x = np.asarray(x, float)
    if x.shape[0] <= 3 * (order + 1):
        return x
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, x, axis=0)


def resample_interp(sig, n):
    """Linear time-normalisation to `n` samples (np.interp; passes through samples, no ringing).

    Works for 1-D (N,) and 2-D (N, k) signals.
    """
    sig = np.asarray(sig, float)
    src = np.linspace(0.0, 1.0, len(sig))
    dst = np.linspace(0.0, 1.0, n)
    if sig.ndim == 1:
        return np.interp(dst, src, sig)
    return np.column_stack([np.interp(dst, src, sig[:, i]) for i in range(sig.shape[1])])


def derivative(x, fs=FS_HZ, cutoff=CUTOFF_HZ, order=ORDER, prefilter=True):
    """d/dt of a signal at the native rate: (optionally) low-pass then central difference.

    Returns an array the same length as `x`. Used for joint velocity (rad/s) and for
    Cartesian velocity/acceleration.
    """
    x = np.asarray(x, float)
    gaps = ~np.isfinite(x)
    if gaps.any():
        # Masked samples (e.g. the shoulder plane of elevation near gimbal lock): bridge them
        # linearly so the zero-phase filter does not spread NaN over the whole signal, then put
        # the NaN back -- a derivative is never reported where the angle itself is not.
        x = fill_gaps(x)
        if x is None:
            return np.full(gaps.shape, np.nan)
    if prefilter:
        x = butter_lowpass(x, cutoff, fs, order)
    d = np.gradient(x, 1.0 / fs, axis=0)
    if gaps.any():
        d = np.where(gaps, np.nan, d)
    return d


def fill_gaps(x):
    """Linear interpolation over non-finite samples, along axis 0 (edges held). None if empty."""
    x = np.array(x, float, copy=True)
    if x.ndim == 1:
        ok = np.isfinite(x)
        if not ok.any():
            return None
        idx = np.arange(x.size)
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
        return x
    cols = [fill_gaps(x[:, j]) for j in range(x.shape[1])]
    if any(c is None for c in cols):
        return None
    return np.column_stack(cols)


def speed(pos, fs=FS_HZ):
    """|v| = ||d p / dt|| for a Cartesian trajectory pos (N, 3)."""
    v = derivative(np.asarray(pos, float), fs)
    return np.linalg.norm(v, axis=1)


def accel_mag(pos, fs=FS_HZ):
    """|a| = ||d^2 p / dt^2|| for a Cartesian trajectory pos (N, 3)."""
    v = derivative(np.asarray(pos, float), fs)
    a = np.gradient(v, 1.0 / fs, axis=0)
    return np.linalg.norm(a, axis=1)
