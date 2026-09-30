"""
backend/dsp.py — Digital Signal Processing module for ICU ECG.
Implements:
1. Moving-average baseline wander removal (Clean) preserving QRS amplitude >= 85%.
2. Adaptive-threshold detector with 200 ms refractory blanking period (Detect).
3. Signal Quality Index (SQI) computation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any

FS_DEFAULT = 250  # Hz default sampling rate


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Fast moving average using cumulative sum with edge handling."""
    n = len(x)
    if window <= 1 or n == 0:
        return x.copy()
    
    window = min(window, n)
    half = window // 2
    cumsum = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(x, out=cumsum[1:])
    
    out = np.empty(n, dtype=np.float32)
    # Vectorized moving average with boundary clamping
    idx_a = np.maximum(0, np.arange(n) - half)
    idx_b = np.minimum(n, np.arange(n) + half + 1)
    counts = idx_b - idx_a
    out = (cumsum[idx_b] - cumsum[idx_a]) / counts
    return out.astype(np.float32)


def clean_baseline(x: np.ndarray, bw: int) -> np.ndarray:
    """
    Moving-average baseline removal.
    Subtracts baseline estimate from raw signal, preserving QRS sharpness.
    """
    baseline = moving_average(x, bw)
    return x - baseline


def detect_rpeaks(y: np.ndarray, sm: int, th: float, mw: int, fs: int = FS_DEFAULT) -> Dict[str, Any]:
    """
    Adaptive-threshold R-peak detector with 200 ms refractory period.
    Steps:
      1. Smooth signal with moving average window `sm`.
      2. Differentiate and square to emphasize high-frequency QRS slope.
      3. Moving window integration with window `mw`.
      4. Adaptive threshold: max(th * P_97, 2.5 * P_50).
      5. Peak search with a strict 200 ms refractory blanking period.
    """
    n = len(y)
    if n < 3:
        return {"pk": [], "energy": np.zeros(n, dtype=np.float32)}

    # Step 1: Smooth
    s = moving_average(y, max(1, int(sm)))
    
    # Step 2: Differentiate & Square
    diff = np.zeros(n, dtype=np.float32)
    diff[2:] = s[2:] - s[:-2]
    energy = diff * diff
    
    # Step 3: Moving Window Integration
    e_int = moving_average(energy, max(1, int(mw)))
    
    # Step 4: Adaptive Threshold
    p97 = float(np.percentile(e_int, 97))
    p50 = float(np.percentile(e_int, 50))
    threshold = max(float(th) * p97, 2.5 * p50)
    
    # Step 5: Peak search with 200 ms refractory blanking
    refractory_samples = int(0.20 * fs)  # 200 ms
    peaks: List[int] = []
    last_peak = -999999
    
    for i in range(1, n - 1):
        if e_int[i] > threshold and e_int[i] >= e_int[i - 1] and e_int[i] > e_int[i + 1]:
            # Refine peak to maximum in original smoothed signal within +- 15 samples
            a = max(0, i - 15)
            b = min(n, i + 16)
            best_idx = a + int(np.argmax(s[a:b]))
            
            if best_idx - last_peak > refractory_samples:
                peaks.append(best_idx)
                last_peak = best_idx
                
    return {"pk": peaks, "threshold": threshold, "energy": e_int}


def compute_sqi(y: np.ndarray) -> float:
    """
    Compute Signal Quality Index (0.0 = noise/invalid, 1.0 = clean clinical ECG).
    Uses relative power of high-frequency noise derivatives vs signal peak envelope.
    """
    n = len(y)
    if n < 5:
        return 1.0
    
    d2 = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
    abs_y = np.abs(y)
    
    sg = float(np.percentile(d2, 50)) / 1.652 + 1e-6
    r = float(np.percentile(abs_y, 98)) / sg
    sqi = (np.log10(max(r, 1e-6)) - 0.4) / 0.9
    return float(np.clip(sqi, 0.0, 1.0))
