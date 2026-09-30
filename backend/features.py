"""
backend/features.py — RR-Feature extraction and Heart Rate Variability (HRV) metrics.
Ported from open-source ECG signal processing algorithms:
  - R-peak detection & RR intervals
  - Heart Rate (HR bpm)
  - SDNN (Standard Deviation of NN intervals in ms)
  - RMSSD (Root Mean Square of Successive Differences in ms)
  - NN50 (Count of successive differences > 50 ms)
  - CV (Coefficient of Variation of RR intervals)
  - Rhythm classification: Normal sinus rhythm, Atrial fibrillation suspected, Other arrhythmia, Bradycardia, Tachycardia
"""

from __future__ import annotations
import math
from typing import Dict, List, Any, Optional
import numpy as np


def analyze_rr_features(signal: np.ndarray, fs: int = 250) -> Dict[str, Any]:
    """
    Computes full RR interval features, HRV metrics, rhythm classification, and clinical advice.
    """
    n = len(signal)
    if n < fs * 2:
        return {
            "p": [],
            "y": signal.tolist() if isinstance(signal, np.ndarray) else signal,
            "rr": [],
            "hr": 0.0,
            "m": 0.0,
            "sd": 0.0,
            "rms": 0.0,
            "nn50": 0,
            "cv": 0.0,
            "rhythm": "Insufficient beats",
            "condition": "Indeterminate",
            "prog": "Need at least 2-3 seconds of ECG recording for analysis.",
            "items": [],
        }

    # Baseline removal using moving average of length fs
    w_base = max(5, int(fs))
    half_base = w_base // 2
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    base = (cumsum[np.minimum(n, np.arange(n) + half_base + 1)] - cumsum[np.maximum(0, np.arange(n) - half_base)]) / (
        np.minimum(n, np.arange(n) + half_base + 1) - np.maximum(0, np.arange(n) - half_base)
    )
    y = signal - base

    # Small smoothing
    w_sm = 5
    half_sm = w_sm // 2
    cumsum_y = np.cumsum(np.insert(y, 0, 0))
    h = (cumsum_y[np.minimum(n, np.arange(n) + half_sm + 1)] - cumsum_y[np.maximum(0, np.arange(n) - half_sm)]) / (
        np.minimum(n, np.arange(n) + half_sm + 1) - np.maximum(0, np.arange(n) - half_sm)
    )

    mx = float(np.max(h))
    threshold = 0.45 * mx
    peaks: List[int] = []
    last = -1e9
    min_dist = 0.25 * fs  # 250 ms minimum distance

    for i in range(1, n - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > threshold and (i - last) > min_dist:
            peaks.append(i)
            last = i

    rr: List[float] = []
    for i in range(1, len(peaks)):
        rr.append((peaks[i] - peaks[i - 1]) / float(fs))

    if len(rr) < 2:
        m = rr[0] if rr else 0.0
        sd = 0.0
        rms = 0.0
        nn50 = 0
        hr = 60.0 / m if m > 0 else 0.0
        cv = 0.0
    else:
        m = float(np.mean(rr))
        sd = float(np.std(rr, ddof=0))
        diffs = np.diff(rr)
        rms = float(np.sqrt(np.mean(diffs * diffs)))
        nn50 = int(np.sum(np.abs(diffs) > 0.050))  # > 50 ms difference
        hr = 60.0 / m if m > 0 else 0.0
        cv = (sd / m) if m > 0 else 0.0

    # Rhythm classification logic
    cond = "Bradycardia" if hr < 60.0 else ("Tachycardia" if hr > 100.0 else "Normal heart rate")
    if len(rr) < 3:
        rhy = "Insufficient beats"
    elif cv > 0.22:
        rhy = "Atrial fibrillation suspected"
    elif cv > 0.09:
        rhy = "Other arrhythmia suspected"
    else:
        rhy = "Normal sinus rhythm"

    # Clinical advice
    if rhy == "Insufficient beats":
        prog = "Could not detect enough beats: check the sampling rate, column and signal quality."
    elif rhy == "Normal sinus rhythm" and 60.0 <= hr <= 100.0:
        prog = "No problem detected"
    elif "Atrial" in rhy or "Other" in rhy:
        prog = "Consult doctor ASAP" if hr < 60.0 else "Doctor review needed"
    else:
        prog = "Heart rate outside normal range: doctor review suggested"

    items = [
        ["Rhythm", rhy],
        ["Condition", cond],
        ["Heart rate", f"{hr:.0f} bpm"],
        ["R peaks", str(len(peaks))],
        ["Avg RR", f"{m:.2f} s"],
        ["HRV (SDNN)", f"{sd * 1000:.0f} ms"],
        ["RMSSD", f"{rms * 1000:.0f} ms"],
        ["NN50", str(nn50)],
        ["RR variation", f"{cv * 100:.0f}%"],
    ]

    return {
        "p": peaks,
        "y": h.tolist(),
        "rr": rr,
        "hr": float(hr),
        "m": float(m),
        "sd": float(sd),
        "rms": float(rms),
        "nn50": int(nn50),
        "cv": float(cv),
        "rhythm": rhy,
        "condition": cond,
        "prog": prog,
        "advice": f"Advice: {prog} (research demo, not a medical device)",
        "items": items,
    }
