"""
backend/fuzzy.py — Fuzzy Logic reasoning engine for ICU arrhythmia classification.
Combines:
  - Heart Rate (HR in bpm)
  - Rhythm Regularity (CV: Coefficient of Variation of RR intervals)
  - Signal Quality (Q / SQI in [0, 1])

Rules:
  - VT: High HR + Regular rhythm
  - VF: Very High / Chaotic HR + Good signal quality
  - Noise: Poor quality + Irregular variation -> Held for clinical review (no false alarm)
  - Alarm: Confidence >= 0.6 and Noise < 0.5
  - Hold: (Confidence < 0.6 and Noise >= 0.5) or (Poor quality and low confidence)
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function with safe boundary handling."""
    if x is None:
        return 0.0
    val_ab = (x - a) / (b - a) if b != a else 1.0
    val_dc = (d - x) / (d - c) if d != c else 1.0
    return float(np.clip(min(val_ab, 1.0, val_dc), 0.0, 1.0))


def evaluate_fuzzy(hr: Optional[float], cv: Optional[float], q: float) -> Dict[str, Any]:
    """
    Evaluates fuzzy membership and inference rules.
    """
    m = {
        "hrHigh": 0.0,
        "hrVery": 0.0,
        "regular": 0.0,
        "irregular": 0.0,
        "good": trapmf(q, 0.35, 0.6, 1.0, 1.01),
        "poor": trapmf(q, -1.0, -0.5, 0.25, 0.5),
    }

    if hr is not None:
        m["hrHigh"] = trapmf(hr, 110.0, 150.0, 400.0, 401.0)
        m["hrVery"] = trapmf(hr, 180.0, 230.0, 400.0, 401.0)
    
    if cv is not None:
        m["regular"] = trapmf(cv, -1.0, -0.5, 0.12, 0.3)
        m["irregular"] = trapmf(cv, 0.15, 0.35, 5.0, 6.0)

    # Fuzzy Rule Consequents
    vt = min(m["hrHigh"], m["regular"])
    vf = min(m["hrVery"], m["good"])
    noise = min(m["poor"], m["irregular"])
    
    confidence = max(vt, vf)
    
    # Alarm Trigger Decision:
    # High confidence and NOT suppressed by noise artifact
    alarm = (confidence >= 0.6) and (noise < 0.5)
    
    # Hold for review: Noise detected or uncertain quality
    hold = (confidence < 0.6 and noise >= 0.5) or (hr is not None and m["poor"] > 0.5 and confidence < 0.6)

    status_label = "LETHAL ALARM" if alarm else ("HOLD FOR REVIEW" if hold else "NORMAL MONITORING")

    return {
        "hr": hr,
        "cv": cv,
        "q": q,
        "memberships": m,
        "vt": vt,
        "vf": vf,
        "noise": noise,
        "confidence": confidence,
        "alarm": bool(alarm),
        "hold": bool(hold),
        "status": status_label,
    }
