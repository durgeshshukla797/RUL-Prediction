"""
utils/preprocessing.py
-----------------------
Reusable preprocessing pipeline for NASA CMAPSS dataset.

All functions are pure (no global state). The scaler is fitted ONLY on
training data and applied to test data via apply_scaler(), preventing
data leakage between train/test splits.

Key design decisions
--------------------
* RUL is normalized to [0, 1] (divided by rul_cap) for training targets.
  Raw-cycle RUL is preserved separately for evaluation / frontend display.
* Constant-variance features (std=0 across entire training set) are dropped
  automatically — this catches e.g. c3 in FD001 which is all-zeros.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Column definitions ──────────────────────────────────────────────────────
COLUMN_NAMES = (
    ["unit", "time", "c1", "c2", "c3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001 — drop them
# c3 also has std=0 in FD001 and is caught by auto-detection below
CONSTANT_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}

# All sensors
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]

# RUL cap — piecewise-linear approach from literature
DEFAULT_RUL_CAP = 125

# Health thresholds (RUL cycles)
HEALTHY_THRESHOLD = 30   # RUL > 30  → Healthy  (class 2)
CRITICAL_THRESHOLD = 15  # RUL <= 15 → Critical (class 0)
# 15 < RUL <= 30 → Warning (class 1)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load a CMAPSS .txt file, apply column names, set dtypes.

    CMAPSS files often have a trailing space at the end of each row.
    With sep=r'\\s+', that creates an extra unnamed NaN column.
    ``usecols=range(len(COLUMN_NAMES))`` discards it before dropna runs.
    ``index_col=False`` prevents the first column being treated as index
    when column counts differ.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        index_col=False,
        usecols=range(len(COLUMN_NAMES)),  # drop any trailing whitespace column
        engine="python",
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=COLUMN_NAMES)   # only drop rows with NaN in known cols
    return df


def load_rul_file(path: str) -> pd.DataFrame:
    """Load RUL_FD00x.txt — one RUL value per engine."""
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    rul["unit"] = rul.index + 1
    return rul


def load_dataset(
    train_path: str,
    test_path: str,
    rul_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and ground-truth RUL files.

    Returns
    -------
    train_df, test_df, rul_df : DataFrames (raw, no preprocessing applied)
    """
    train_df = load_raw(train_path)
    test_df  = load_raw(test_path)
    rul_df   = load_rul_file(rul_path)
    return train_df, test_df, rul_df


# ── RUL / Label computation ──────────────────────────────────────────────────

def add_rul_train(df: pd.DataFrame, rul_cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """
    Compute RUL for training data.
    rul_raw = min(max_cycle - current_cycle, rul_cap)   [in cycles]
    rul     = rul_raw / rul_cap                          [normalized 0-1]
    Both columns are stored so evaluation can denormalize.
    """
    df = df.copy()
    max_cycles  = df.groupby("unit")["time"].transform("max")
    rul_raw     = (max_cycles - df["time"]).clip(upper=rul_cap)
    df["rul_raw"] = rul_raw.astype(np.float32)
    df["rul"]     = (rul_raw / rul_cap).astype(np.float32)   # normalized
    return df


def add_rul_test(
    df: pd.DataFrame, rul_df: pd.DataFrame, rul_cap: int = DEFAULT_RUL_CAP
) -> pd.DataFrame:
    """
    Attach ground-truth RUL to test data.
    The RUL file gives the RUL at the LAST cycle of each engine.
    We back-calculate from there.
    Both raw and normalized columns stored.
    """
    df        = df.copy()
    last_cycle = df.groupby("unit")["time"].transform("max")
    df         = df.merge(rul_df[["unit", "rul"]], on="unit", how="left")
    # rul at each row = ground-truth_rul + (max_cycle - current_cycle), capped
    rul_raw    = (df["rul"] + (last_cycle - df["time"])).clip(upper=rul_cap)
    df["rul_raw"] = rul_raw.astype(np.float32)
    df["rul"]     = (rul_raw / rul_cap).astype(np.float32)   # normalized
    return df


def add_health_label(
    df: pd.DataFrame,
    healthy_threshold: int = HEALTHY_THRESHOLD,
    critical_threshold: int = CRITICAL_THRESHOLD,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> pd.DataFrame:
    """
    Derive 3-class health label from rul_raw column.

    Classes
    -------
    0 : Critical  (RUL_raw <= critical_threshold)
    1 : Warning   (critical_threshold < RUL_raw <= healthy_threshold)
    2 : Healthy   (RUL_raw > healthy_threshold)
    """
    df      = df.copy()
    rul_raw = df["rul_raw"]
    label   = np.full(len(df), 2, dtype=np.int32)       # Healthy default
    label[rul_raw <= healthy_threshold]  = 1             # Warning
    label[rul_raw <= critical_threshold] = 0             # Critical
    df["label"] = label
    return df


# ── Feature selection ─────────────────────────────────────────────────────────

def get_feature_columns(
    train_df: pd.DataFrame,
    drop_constant: bool = True,
    constant_sensors: set | None = None,
    std_threshold: float = 1e-4,
) -> list[str]:
    """
    Return the list of sensor + operational-condition columns to use as features.

    Drops:
    - Explicitly listed constant sensors (CONSTANT_SENSORS set)
    - Any column whose std across the training set is below std_threshold
      (catches c3 in FD001 which is all-zeros)
    """
    if constant_sensors is None:
        constant_sensors = CONSTANT_SENSORS

    candidate_cols = [
        col for col in COLUMN_NAMES[2:]  # skip 'unit', 'time'
        if col not in ("rul", "rul_raw", "label")
        and col in train_df.columns
        and col not in constant_sensors
    ]

    if drop_constant:
        # Auto-detect zero-variance columns
        stds = train_df[candidate_cols].std()
        candidate_cols = [c for c in candidate_cols if stds[c] > std_threshold]

    return candidate_cols


# ── Smoothing ─────────────────────────────────────────────────────────────────

def apply_ema_smoothing(df: pd.DataFrame, feature_cols: list[str], alpha: float = 0.3) -> pd.DataFrame:
    """Apply Exponential Moving Average smoothing group-wise per engine unit on sensor features."""
    df = df.copy()
    if "unit" in df.columns:
        def smooth_group(g):
            g_copy = g.copy()
            g_copy[feature_cols] = g_copy[feature_cols].ewm(alpha=alpha, adjust=False).mean()
            return g_copy
        df = df.groupby("unit", group_keys=False).apply(smooth_group)
    else:
        df[feature_cols] = df[feature_cols].ewm(alpha=alpha, adjust=False).mean()
    return df


# ── Normalization ─────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    """Fit a MinMaxScaler on training data. Returns fitted scaler."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list[str]
) -> pd.DataFrame:
    """Apply a pre-fitted scaler. Returns a new DataFrame."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_train(
    train_df: pd.DataFrame,
    rul_cap: int = DEFAULT_RUL_CAP,
    drop_constant: bool = True,
) -> tuple[pd.DataFrame, MinMaxScaler, list[str]]:
    """
    Full preprocessing pipeline for training data.

    Returns
    -------
    df           : preprocessed DataFrame with 'rul' (normalized), 'rul_raw',
                   and 'label' columns
    scaler       : fitted MinMaxScaler (fitted ONLY on training features)
    feature_cols : list of feature column names (constant cols removed)
    """
    df           = add_rul_train(train_df, rul_cap=rul_cap)
    df           = add_health_label(df, rul_cap=rul_cap)
    feature_cols = get_feature_columns(df, drop_constant=drop_constant)
    df           = apply_ema_smoothing(df, feature_cols, alpha=0.3)
    scaler       = fit_scaler(df, feature_cols)
    df           = apply_scaler(df, scaler, feature_cols)

    # Debug print
    print(f"[preprocess] Train rows: {len(df)}")
    print(f"[preprocess] Features ({len(feature_cols)}): {feature_cols}")
    rul_raw_vals = df["rul_raw"].values
    print(f"[preprocess] RUL_raw range: [{rul_raw_vals.min():.0f}, {rul_raw_vals.max():.0f}]  "
          f"mean={rul_raw_vals.mean():.1f}")
    rul_norm_vals = df["rul"].values
    print(f"[preprocess] RUL_norm range: [{rul_norm_vals.min():.3f}, {rul_norm_vals.max():.3f}]  "
          f"mean={rul_norm_vals.mean():.3f}  (training target)")
    counts = {v: int((df["label"] == v).sum()) for v in [0, 1, 2]}
    total = sum(counts.values())
    names = {0: "Critical", 1: "Warning", 2: "Healthy"}
    print("[preprocess] Label distribution:")
    for cls in [0, 1, 2]:
        print(f"             Class {cls} ({names[cls]:8s}): "
              f"{counts[cls]:6d}  ({100*counts[cls]/total:.1f}%)")

    return df, scaler, feature_cols


def preprocess_test(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: list[str],
    rul_cap: int = DEFAULT_RUL_CAP,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for test data using a pre-fitted scaler.
    """
    df = add_rul_test(test_df, rul_df, rul_cap=rul_cap)
    df = add_health_label(df, rul_cap=rul_cap)
    df = apply_ema_smoothing(df, feature_cols, alpha=0.3)
    df = apply_scaler(df, scaler, feature_cols)
    return df


# ── Artifact persistence ──────────────────────────────────────────────────────

def save_preprocessing_artifacts(
    scaler: MinMaxScaler,
    feature_cols: list[str],
    output_dir: str,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> None:
    """Save scaler, feature column list, and rul_cap to disk."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    meta = {"feature_cols": feature_cols, "rul_cap": rul_cap}
    with open(os.path.join(output_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, "rul_cap.json"), "w") as f:
        json.dump({"rul_cap": rul_cap}, f)
    print(f"[preprocess] Saved scaler & feature_cols to '{output_dir}/'")


def load_preprocessing_artifacts(
    artifacts_dir: str,
) -> tuple[MinMaxScaler, list[str], int]:
    """Load scaler, feature column list, and rul_cap from disk."""
    with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(artifacts_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    rul_cap_path = os.path.join(artifacts_dir, "rul_cap.json")
    rul_cap = 125  # default
    if os.path.exists(rul_cap_path):
        with open(rul_cap_path) as f:
            rul_cap = json.load(f)["rul_cap"]
    return scaler, feature_cols, rul_cap


# ── Inference helper ──────────────────────────────────────────────────────────

def preprocess_engine_df(
    raw_engine_df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Preprocess a single-engine DataFrame for real-time inference.
    Does NOT compute RUL (unknown at inference time).
    """
    df = raw_engine_df.copy()
    df = apply_ema_smoothing(df, feature_cols, alpha=0.3)
    df = apply_scaler(df, scaler, feature_cols)
    return df
