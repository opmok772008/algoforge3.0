"""
backend/app.py — FastAPI Application for Fuzzy-Evolutionary ICU Arrhythmia Pipeline.
Serves both the REST API and the interactive website.
"""

from __future__ import annotations
import asyncio
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.dsp import clean_baseline, detect_rpeaks, compute_sqi
from backend.fuzzy import evaluate_fuzzy
from backend.synthetic import ECGSimulator
from backend.ga import DEFAULT_PARAMS, GeneticAlgorithm
from backend.features import analyze_rr_features
from backend.validation import StreamingMonitor, run_all_validations

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = BASE_DIR / "index.html"
FRONTEND_HTML = BASE_DIR / "FRONTEND.html"

app = FastAPI(
    title="Fuzzy-Evolutionary ICU Arrhythmia Pipeline API",
    description="Backend for real-time noise-resilient arrhythmia detection, fuzzy inference, and GA optimization",
    version="1.0.0",
)

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global streaming monitor instance and active parameters
active_params: Dict[str, Any] = dict(DEFAULT_PARAMS)
global_monitor = StreamingMonitor(seed=42, params=active_params)
active_jobs: Dict[str, Dict[str, Any]] = {}


# --- Request & Response Models ---
class ParamsModel(BaseModel):
    bw: int = Field(200, ge=50, le=500, description="Baseline window")
    sm: int = Field(5, ge=1, le=15, description="Smoothing window")
    th: float = Field(0.30, ge=0.05, le=0.90, description="Adaptive threshold factor")
    mw: int = Field(25, ge=8, le=50, description="Integration window")


class StreamRequest(BaseModel):
    scenario: str = Field("normal", description="Rhythm: normal, vt, vf")
    wander: bool = Field(False, description="Baseline wander interference")
    motion: bool = Field(False, description="Motion artifact interference")
    emg: bool = Field(False, description="Severe EMG burst interference")


class AnalyzeRequest(BaseModel):
    samples: List[float]
    fs: Optional[int] = 250


class OptimizeRequest(BaseModel):
    iterations: Optional[int] = 20
    seed: Optional[int] = 5


# --- Root & Static Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    if FRONTEND_HTML.exists():
        return FileResponse(FRONTEND_HTML)
    return HTMLResponse("<h1>Fuzzy-Evolutionary ICU Pipeline API is active</h1>")


@app.get("/FRONTEND.html", response_class=HTMLResponse)
async def serve_frontend_html():
    if FRONTEND_HTML.exists():
        return FileResponse(FRONTEND_HTML)
    return FileResponse(INDEX_HTML)


@app.get("/sample_ecg.csv")
async def serve_sample_csv():
    sample_path = BASE_DIR / "sample_ecg.csv"
    if sample_path.exists():
        return FileResponse(sample_path, media_type="text/csv", filename="sample_ecg.csv")
    raise HTTPException(status_code=404, detail="Sample CSV not found")


# --- Health & Diagnostic Endpoints ---
@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "Fuzzy-Evolutionary ICU Arrhythmia Pipeline",
        "backend": "FastAPI / Python",
        "version": "1.0.0",
        "active_params": active_params,
        "streaming_buffer_bytes": int(global_monitor.buffer.nbytes),
        "timestamp": time.time(),
    }


# --- Parameter Management Endpoints ---
@app.get("/api/params")
async def get_params():
    return {
        "default": DEFAULT_PARAMS,
        "active": active_params,
    }


@app.post("/api/params")
async def set_params(params: ParamsModel):
    global active_params, global_monitor
    active_params = params.model_dump()
    global_monitor.params = active_params
    return {"status": "updated", "params": active_params}


# --- Live Streaming Pipeline Endpoint ---
@app.post("/api/stream")
async def stream_tick(req: StreamRequest):
    """
    Simulates one real-time streaming time-step (25 samples = 100 ms).
    Cleans signal, detects R-peaks with 200 ms refractory blanking, evaluates SQI,
    applies fuzzy rules, and returns live status.
    """
    global global_monitor
    global_monitor.set_scenario(
        scenario=req.scenario,
        wander=req.wander,
        motion=req.motion,
        emg=req.emg
    )
    global_monitor.step(chunk_samples=25)

    sub_raw = global_monitor.buffer[-global_monitor.window_size:]
    cleaned = global_monitor.cleaned_trace
    peaks = global_monitor.detected_peaks
    st = global_monitor.current_state
    lat = global_monitor.get_latency()

    return {
        "raw": sub_raw.tolist(),
        "cleaned": cleaned.tolist(),
        "peaks": peaks,
        "state": st,
        "latency_sec": lat,
        "buffer_size_samples": len(global_monitor.buffer),
        "buffer_bytes": int(global_monitor.buffer.nbytes),
    }


# --- RR-Feature Analysis & ECG Upload Endpoints ---
@app.post("/api/analyze")
async def analyze_ecg(req: AnalyzeRequest):
    """
    Full RR-interval feature extraction, HRV metrics, rhythm classification, and advice.
    """
    signal = np.array(req.samples, dtype=np.float32)
    fs = req.fs or 250
    result = analyze_rr_features(signal, fs=fs)
    return result


@app.post("/api/upload")
async def upload_ecg_file(file: UploadFile = File(...), fs: int = 300, column_idx: int = 0):
    """
    Accepts CSV, TSV, or TXT file upload, parses numeric columns, and performs analysis.
    """
    contents = await file.read()
    text = contents.decode("utf-8", errors="replace")
    
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
        if parts:
            rows.append(parts)

    if not rows:
        raise HTTPException(status_code=400, detail="No readable content found in file")

    # Check for header
    has_header = any(not part.replace(".", "", 1).replace("-", "", 1).isdigit() for part in rows[0])
    headers = rows.pop(0) if has_header else [f"Column {i+1}" for i in range(len(rows[0]))]

    columns = [[] for _ in range(len(headers))]
    for r in rows:
        for idx in range(min(len(r), len(columns))):
            try:
                val = float(r[idx])
                columns[idx].append(val)
            except ValueError:
                pass

    if column_idx >= len(columns) or not columns[column_idx]:
        col_to_use = 0
        for i, c in enumerate(columns):
            if len(c) > 0:
                col_to_use = i
                break
    else:
        col_to_use = column_idx

    selected_data = np.array(columns[col_to_use][:fs * 30], dtype=np.float32)
    if len(selected_data) < fs * 2:
        raise HTTPException(status_code=400, detail="Need at least 2-3 seconds of numeric data")

    analysis = analyze_rr_features(selected_data, fs=fs)
    analysis["column_name"] = headers[col_to_use] if col_to_use < len(headers) else f"Column {col_to_use+1}"
    analysis["available_columns"] = headers
    analysis["total_samples"] = len(selected_data)
    analysis["duration_sec"] = len(selected_data) / fs
    return analysis


# --- Genetic Algorithm Optimization Endpoints ---
def _run_ga_job(job_id: str, iterations: int, seed: int):
    ga = GeneticAlgorithm(seed=seed)
    active_jobs[job_id]["convergence"].append({
        "iter": 0,
        "gbest": float(ga.best["fitness"]),
        "mean": float(ga.mean_fitness),
        "params": {
            "bw": ga.best["bw"],
            "sm": ga.best["sm"],
            "th": round(ga.best["th"], 2),
            "mw": ga.best["mw"],
        },
    })

    for step_num in range(1, iterations + 1):
        step_res = ga.step()
        active_jobs[job_id]["convergence"].append({
            "iter": step_num,
            "gbest": float(step_res["best_fitness"]),
            "mean": float(step_res["mean_fitness"]),
            "params": step_res["best_params"],
        })
        time.sleep(0.02)  # Yield for responsiveness

    active_jobs[job_id]["status"] = "done"
    active_jobs[job_id]["best_params"] = {
        "bw": ga.best["bw"],
        "sm": ga.best["sm"],
        "th": round(ga.best["th"], 2),
        "mw": ga.best["mw"],
        "fitness": round(ga.best["fitness"], 4),
    }


@app.post("/api/optimize")
async def start_optimization(req: OptimizeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "convergence": [],
        "best_params": None,
    }
    background_tasks.add_task(_run_ga_job, job_id, req.iterations or 20, req.seed or 5)
    return {"job_id": job_id, "status": "running"}


@app.get("/api/optimize/{job_id}")
async def get_optimization_status(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]


# --- Validation Suite Endpoints ---
@app.get("/api/validation")
@app.post("/api/validation")
@app.get("/api/tests")
@app.post("/api/tests")
async def run_tests_endpoint():
    """Runs all 8 problem statement verification tests."""
    results = run_all_validations()
    return results


# Mount root directory for any additional static resources
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")
