"""
utils/windowing.py
------------------
Sliding-window (sequence) generation for time-series inputs.

Works identically for training, validation, and inference.

Target columns
--------------
'rul'     : normalized RUL in [0, 1]  — used as model training target
'rul_raw' : raw RUL in cycles         — used for RMSE/MAE evaluation (denormalized)
'label'   : 0/1/2 health class        — classification target
"""

import numpy as np
import pandas as pd


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate overlapping time windows from a multi-engine DataFrame.

    Parameters
    ----------
    df           : preprocessed DataFrame (unit, time, features, rul, rul_raw, label)
    feature_cols : ordered list of feature column names
    window_size  : number of time steps per window
    stride       : step between consecutive window starts

    Returns
    -------
    X       : (N, window_size, n_features) float32 — sensor sequences
    y_reg   : (N,) float32 — normalized RUL [0,1] at last timestep  (model target)
    y_reg_raw: (N,) float32 — raw RUL [cycles] at last timestep     (for evaluation)
    y_clf   : (N,) int32   — health class at last timestep
    """
    X_list, y_reg_list, y_raw_list, y_clf_list = [], [], [], []

    has_rul     = "rul"     in df.columns
    has_rul_raw = "rul_raw" in df.columns
    has_label   = "label"   in df.columns

    for engine_id in df["unit"].unique():
        engine_df = df[df["unit"] == engine_id].sort_values("time")
        n = len(engine_df)

        for start in range(0, n - window_size + 1, stride):
            end    = start + window_size
            window = engine_df.iloc[start:end]
            X_list.append(window[feature_cols].values.astype(np.float32))
            if has_rul:
                y_reg_list.append(float(window["rul"].iloc[-1]))
            if has_rul_raw:
                y_raw_list.append(float(window["rul_raw"].iloc[-1]))
            if has_label:
                y_clf_list.append(int(window["label"].iloc[-1]))

    X       = np.array(X_list,   dtype=np.float32)
    y_reg   = np.array(y_reg_list,  dtype=np.float32) if y_reg_list else np.array([])
    y_raw   = np.array(y_raw_list,  dtype=np.float32) if y_raw_list else np.array([])
    y_clf   = np.array(y_clf_list,  dtype=np.int32)   if y_clf_list else np.array([])

    return X, y_reg, y_raw, y_clf


def create_sequences_test_last(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For test data, extract:
    - ONLY the last window per engine for regression evaluation.
    - All windows for classification evaluation.

    Returns
    -------
    X_reg      : (n_engines, window_size, n_features)
    y_reg      : (n_engines,) normalized RUL
    y_reg_raw  : (n_engines,) raw RUL in cycles
    X_clf      : (N_total, window_size, n_features)
    y_clf      : (N_total,) health class
    """
    X_reg_list, y_reg_list, y_raw_list = [], [], []
    X_clf_list, y_clf_list = [], []

    for engine_id in sorted(df["unit"].unique()):
        engine_df = df[df["unit"] == engine_id].sort_values("time")
        n = len(engine_df)

        for start in range(0, n - window_size + 1, 1):
            end    = start + window_size
            window = engine_df.iloc[start:end]
            x      = window[feature_cols].values.astype(np.float32)
            X_clf_list.append(x)
            y_clf_list.append(int(window["label"].iloc[-1]))

            if start == n - window_size:           # last window = regression test point
                X_reg_list.append(x)
                y_reg_list.append(float(window["rul"].iloc[-1]))
                y_raw_list.append(float(window["rul_raw"].iloc[-1]))

    X_reg    = np.array(X_reg_list, dtype=np.float32)
    y_reg    = np.array(y_reg_list, dtype=np.float32)
    y_raw    = np.array(y_raw_list, dtype=np.float32)
    X_clf    = np.array(X_clf_list, dtype=np.float32)
    y_clf    = np.array(y_clf_list, dtype=np.int32)

    return X_reg, y_reg, y_raw, X_clf, y_clf


def create_inference_sequence(
    engine_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
) -> np.ndarray | None:
    """
    Create a single inference window from the last `window_size` rows of an engine.

    If the engine has fewer rows than `window_size`, the sequence is front-padded
    by repeating the earliest available row.  This means even brand-new engines
    with only a handful of cycles can still receive a prediction.

    Returns
    -------
    X : (1, window_size, n_features)  — never None
    """
    engine_df = engine_df.sort_values("time")
    data = engine_df[feature_cols].values.astype(np.float32)

    if len(data) >= window_size:
        # Normal case: take the last window_size rows
        window = data[-window_size:]
    else:
        # Short engine: front-pad with the first row repeated
        pad_rows = window_size - len(data)
        pad = np.tile(data[[0]], (pad_rows, 1))   # repeat first row
        window = np.vstack([pad, data])

    return window[np.newaxis, ...]  # (1, window_size, n_features)
