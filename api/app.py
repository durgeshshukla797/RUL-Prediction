"""
api/app.py
----------
FastAPI backend for the CMAPSS Predictive Maintenance System.
Supports Multi-Dataset lazy loading and Static File mounts for plots.
"""

import os
import sys
import json
import io
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from utils.preprocessing import (
    load_preprocessing_artifacts, preprocess_engine_df, apply_ema_smoothing,
    add_health_label, apply_scaler,
    COLUMN_NAMES, HEALTHY_THRESHOLD, CRITICAL_THRESHOLD, DEFAULT_RUL_CAP,
)
from utils.windowing import create_inference_sequence, create_sequences

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
HEALTH_LABELS = {0: "Critical", 1: "Warning", 2: "Healthy"}
HEALTH_COLORS = {0: "critical", 1: "warning", 2: "healthy"}
DEFAULT_WINDOW = 30
RUL_CAP = 125

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CMAPSS Multi-Dataset API",
    description="Remaining Useful Life prediction with dynamic dataset loading",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# ── Runtime state (Lazy Loaded) ───────────────────────────────────────────────
dataset_state: dict = {}

def get_state(dataset: str):
    dataset = dataset.upper()
    if dataset in dataset_state:
        return dataset_state[dataset]
        
    d_path = os.path.join(MODELS_DIR, dataset)
    if not os.path.exists(d_path):
        raise HTTPException(404, f"Dataset models {dataset} not found. Run train.py first.")
        
    print(f"[api] Lazy loading dataset {dataset} into memory...")
    st = {}
    try:
        st["scaler"], st["feature_cols"], st["rul_cap"] = load_preprocessing_artifacts(d_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load artifacts for {dataset}: {e}")
        
    st["window_size"] = DEFAULT_WINDOW
    
    best_model_path = os.path.join(d_path, "best_model.keras")
    if os.path.exists(best_model_path):
        st["primary_model"] = tf.keras.models.load_model(best_model_path)
    else:
        raise HTTPException(503, f"best_model.keras not found for {dataset}")
        
    for key, filename in [
        ("metrics", "metrics.json"),
        ("history", "training_history.json"),
        ("predictions", "test_predictions.json"),
        ("engine_data", "engine_data.json"),
        ("metadata", "metadata.json"),
        ("best_model_info", "best_model_info.json"),
    ]:
        p = os.path.join(d_path, filename)
        st[key] = json.load(open(p)) if os.path.exists(p) else {}
        
    dataset_state[dataset] = st
    return st


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    engine_id: int = Field(..., description="Engine unit number (1-indexed)")
    cycle: int = Field(..., ge=1, description="Current cycle number")
    dataset: str = Field("FD001", description="Dataset to use")


class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float
    health_class: int
    health_label: str
    health_color: str
    confidence: float
    model_used: str


# ── Helper: run inference on a sequence ──────────────────────────────────────

def _predict_sequence(X: np.ndarray, model, model_name: str, rul_cap: int) -> dict:
    rul_pred_norm, health_proba = model.predict(X, verbose=0)
    rul_raw = max(0.0, float(rul_pred_norm[0, 0]) * rul_cap)
    health_class = int(np.argmax(health_proba[0]))
    confidence   = float(np.max(health_proba[0]))
    return {
        "predicted_rul": round(rul_raw, 2),
        "health_class":  health_class,
        "health_label":  HEALTH_LABELS[health_class],
        "health_color":  HEALTH_COLORS[health_class],
        "confidence":    round(confidence, 4),
        "model_used":    model_name,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "loaded_datasets": list(dataset_state.keys())}


@app.get("/engines", tags=["data"])
async def list_engines(dataset: str = Query("FD001")):
    st = get_state(dataset)
    engine_data = st.get("engine_data", {})
    if not engine_data: raise HTTPException(503, "Engine data empty.")
    ids = sorted([int(k) for k in engine_data.keys()])
    return {"engine_ids": ids, "count": len(ids)}


@app.get("/engine/{engine_id}", tags=["data"])
async def get_engine_data(engine_id: int, dataset: str = Query("FD001")):
    st = get_state(dataset)
    engine_data = st.get("engine_data", {})
    key = str(engine_id)
    if key not in engine_data: raise HTTPException(404, f"Engine {engine_id} not found.")
    data = engine_data[key]

    preds = st.get("predictions", [])
    actual_rul, predicted_rul = None, None
    for p in preds:
        if p["engine_id"] == engine_id:
            actual_rul = p["actual_rul"]
            predicted_rul = p["predicted_rul"]
            break

    return {
        "engine_id": engine_id,
        "n_cycles": len(data["cycles"]),
        "cycles": data["cycles"],
        "features": list(st["feature_cols"]),
        "sensors": data["sensors"],
        "actual_rul": actual_rul,
        "predicted_rul": predicted_rul,
    }


@app.get("/engine/{engine_id}/predict/{cycle}", tags=["prediction"])
async def predict_engine_cycle(engine_id: int, cycle: int, dataset: str = Query("FD001")):
    st = get_state(dataset)
    engine_data = st.get("engine_data", {})
    key = str(engine_id)
    if key not in engine_data: raise HTTPException(404, f"Engine {engine_id} not found.")

    feature_cols = st["feature_cols"]
    window_size = st["window_size"]
    eng = engine_data[key]
    all_cycles = eng["cycles"]
    
    valid_cycles = [c for c in all_cycles if c <= cycle]
    if not valid_cycles: raise HTTPException(400, f"Cycle {cycle} before engine start.")

    rows = []
    for c in valid_cycles:
        idx = all_cycles.index(c)
        row = {"unit": engine_id, "time": c}
        for feat in feature_cols: row[feat] = eng["sensors"][feat][idx]
        rows.append(row)

    eng_df = pd.DataFrame(rows)
    if len(eng_df) < window_size:
        raise HTTPException(400, f"Need >= {window_size} cycles.")

    X = create_inference_sequence(eng_df, feature_cols, window_size)
    m = st["primary_model"]
    model_name = st.get("best_model_info", {}).get("best_model", "best_model")
    
    result = _predict_sequence(X, m, model_name, st["rul_cap"])
    return {"engine_id": engine_id, "cycle": cycle, **result}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: PredictRequest):
    st = get_state(request.dataset)
    engine_data = st.get("engine_data", {})
    key = str(request.engine_id)
    if key not in engine_data: raise HTTPException(404, f"Engine {request.engine_id} not found.")

    feature_cols = st["feature_cols"]
    window_size = st["window_size"]
    eng = engine_data[key]
    all_cycles = eng["cycles"]
    
    valid_cycles = [c for c in all_cycles if c <= request.cycle]
    if not valid_cycles: raise HTTPException(400, f"No data for cycle <= {request.cycle}.")

    rows = []
    for c in valid_cycles:
        idx = all_cycles.index(c)
        row = {"unit": request.engine_id, "time": c}
        for feat in feature_cols: row[feat] = eng["sensors"][feat][idx]
        rows.append(row)

    eng_df = pd.DataFrame(rows)
    if len(eng_df) < window_size:
        raise HTTPException(400, f"Need >= {window_size} cycles.")

    X = create_inference_sequence(eng_df, feature_cols, window_size)
    m = st["primary_model"]
    model_name = st.get("best_model_info", {}).get("best_model", "best_model")

    result = _predict_sequence(X, m, model_name, st["rul_cap"])
    return PredictionResponse(
        engine_id=request.engine_id, cycle=request.cycle, **result
    )


@app.post("/upload", tags=["prediction"])
async def upload_csv(file: UploadFile = File(...), dataset: str = Query("FD001")):
    st = get_state(dataset)
    content = await file.read()
    try:
        try:
            text = content.decode(errors="replace")
            df = pd.read_csv(
                io.StringIO(text), sep=r"\s+", header=None,
                names=COLUMN_NAMES, index_col=False,
                usecols=range(len(COLUMN_NAMES)), engine="python"
            )
        except Exception as e:
            raise HTTPException(400, f"Could not parse file: {e}.")

        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=COLUMN_NAMES)
        if df.empty: raise HTTPException(400, "No valid numeric rows.")

        feature_cols = st["feature_cols"]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing: raise HTTPException(400, f"Missing columns: {missing}.")

        # Ema smoothing and apply scaler
        df_norm = apply_ema_smoothing(df, feature_cols, alpha=0.3)
        df_norm = apply_scaler(df_norm, st["scaler"], feature_cols)

        window_size = st["window_size"]
        m = st["primary_model"]
        model_name = st.get("best_model_info", {}).get("best_model", "best_model")

        results = []
        for engine_id in sorted(df_norm["unit"].unique()):
            eng_df = df_norm[df_norm["unit"] == engine_id].sort_values("time")
            X = create_inference_sequence(eng_df, feature_cols, window_size)
            if X is None:
                results.append({"engine_id": int(engine_id), "error": "Insufficient data"})
                continue
            pred = _predict_sequence(X, m, model_name, st["rul_cap"])
            results.append({"engine_id": int(engine_id), "n_cycles": len(eng_df), **pred})

        return {"filename": file.filename, "n_engines": len(results), "predictions": results}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


@app.get("/metrics", tags=["evaluation"])
async def get_metrics(dataset: str = Query("FD001")):
    return get_state(dataset).get("metrics", {})


@app.get("/history", tags=["evaluation"])
async def get_history(dataset: str = Query("FD001")):
    return get_state(dataset).get("history", [])


@app.get("/predictions", tags=["evaluation"])
async def get_predictions(dataset: str = Query("FD001")):
    return get_state(dataset).get("predictions", [])


@app.get("/best_model_info", tags=["evaluation"])
async def get_best_model_info(dataset: str = Query("FD001")):
    return get_state(dataset).get("best_model_info", {})
