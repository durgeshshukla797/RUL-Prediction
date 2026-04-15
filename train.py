"""
train.py  (Multi-Dataset + Model Selection + Plotting)
------------------------------------------------------
End-to-end training script for the CMAPSS predictive maintenance system.

Usage
-----
python train.py --dataset all --epochs 30
python train.py --dataset FD002 --model hybrid

Outputs (saved per dataset)
---------------------------
models/{dataset}/best_model.keras
models/{dataset}/scaler.pkl
models/{dataset}/feature_cols.json
models/{dataset}/metrics.json
models/{dataset}/best_model_info.json
...
plots/{dataset}/loss_curve.png
plots/{dataset}/prediction_vs_actual.png
plots/{dataset}/error_distribution.png
plots/{dataset}/model_comparison.png
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import (
    load_dataset, preprocess_train, preprocess_test,
    save_preprocessing_artifacts, DEFAULT_RUL_CAP,
)
from utils.windowing import create_sequences, create_sequences_test_last
from models_arch.cnn_lstm import build_hybrid_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_FILES = {
    "FD001": ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"),
    "FD002": ("train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt"),
    "FD003": ("train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt"),
    "FD004": ("train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"),
}
HEALTH_LABELS  = {0: "Critical", 1: "Warning", 2: "Healthy"}
HEALTH_COLORS  = {0: "critical", 1: "warning",  2: "healthy"}


# ── Compilation & Callbacks ───────────────────────────────────────────────────

def compile_model(model, learning_rate: float = 5e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss={"rul": "mse", "health": "sparse_categorical_crossentropy"},
        # Increased RUL weight — regression is the primary objective
        loss_weights={"rul": 1.5, "health": 0.3},
        metrics={"rul": ["mae"], "health": ["accuracy"]},
    )
    return model

def get_callbacks(patience: int = 12):
    return [
        EarlyStopping(
            monitor="val_rul_mae",           # monitor regression MAE directly
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_rul_mae",
            factor=0.5,
            patience=max(4, patience // 3),
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
    ]

def make_class_weights(y_clf: np.ndarray) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_clf)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    return cw

def build_sample_weight(y_clf: np.ndarray, class_weights: dict) -> np.ndarray:
    return np.array([class_weights[int(c)] for c in y_clf], dtype=np.float32)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model, X_reg, y_reg_raw, X_clf, y_clf, rul_cap: int,
) -> dict:
    rul_pred_norm, _ = model.predict(X_reg, verbose=0)
    rul_pred_raw     = rul_pred_norm.squeeze() * rul_cap
    rmse = float(np.sqrt(mean_squared_error(y_reg_raw, rul_pred_raw)))
    mae  = float(mean_absolute_error(y_reg_raw, rul_pred_raw))

    _, health_proba = model.predict(X_clf, verbose=0, batch_size=512)
    health_pred = np.argmax(health_proba, axis=1)
    acc = float(accuracy_score(y_clf, health_pred))
    f1  = float(f1_score(y_clf, health_pred, average="weighted", zero_division=0))

    return {
        "rmse": round(rmse, 3), "mae":  round(mae,  3),
        "accuracy": round(acc, 4), "f1": round(f1,  4),
    }

def get_per_engine_predictions(
    model, test_df, feature_cols, window_size: int, rul_cap: int, model_name: str
) -> list[dict]:
    results = []
    for engine_id in sorted(test_df["unit"].unique()):
        edf = test_df[test_df["unit"] == engine_id].sort_values("time")
        if len(edf) < window_size: continue
        X = edf[feature_cols].values[-window_size:].astype(np.float32)[np.newaxis, ...]
        rul_norm, hprob = model.predict(X, verbose=0)
        rul_raw_pred    = float(rul_norm[0, 0]) * rul_cap
        health_class    = int(np.argmax(hprob[0]))
        results.append({
            "engine_id": int(engine_id),
            "actual_rul": round(float(edf["rul_raw"].iloc[-1]), 2),
            "predicted_rul": round(rul_raw_pred, 2),
            "health_class": health_class,
            "health_label": HEALTH_LABELS[health_class],
            "health_color": HEALTH_COLORS[health_class],
            "confidence": round(float(np.max(hprob[0])), 4),
            "model_used": model_name
        })
    return results

def serialize_history(history, model_name: str) -> dict:
    h = history.history
    return {
        "model": model_name,
        "epochs": list(range(1, len(h["loss"]) + 1)),
        "train_loss": [round(v, 6) for v in h["loss"]],
        "val_loss": [round(v, 6) for v in h["val_loss"]],
    }

def export_engine_data(test_df, feature_cols) -> dict:
    data = {}
    for eid in sorted(test_df["unit"].unique()):
        edf = test_df[test_df["unit"] == eid].sort_values("time")
        data[str(int(eid))] = {
            "cycles": [int(c) for c in edf["time"].tolist()],
            "sensors": { col: [round(float(v), 5) for v in edf[col].tolist()] for col in feature_cols },
        }
    return data


# ── Plotting Utilities ────────────────────────────────────────────────────────

def generate_plots(dataset: str, all_histories: list, best_preds: list, all_metrics: dict, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Loss Curve Setup
    try:
        plt.figure(figsize=(10, 6))
        for h in all_histories:
            plt.plot(h["epochs"], h["train_loss"], label=f"{h['model']} (Train)")
            plt.plot(h["epochs"], h["val_loss"], linestyle="--", label=f"{h['model']} (Val)")
        plt.title(f"{dataset} - Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Failed to plot loss_curve: {e}")

    # 2. Prediction vs Actual RUL
    try:
        if best_preds:
            plt.figure(figsize=(12, 6))
            preds_sorted = sorted(best_preds, key=lambda x: x["actual_rul"])
            idx = range(len(preds_sorted))
            actuals = [p["actual_rul"] for p in preds_sorted]
            predicteds = [p["predicted_rul"] for p in preds_sorted]
            plt.plot(idx, actuals, label="Actual RUL", color="blue", linewidth=2)
            plt.plot(idx, predicteds, label="Predicted RUL", color="red", linestyle="--", alpha=0.8)
            plt.fill_between(idx, actuals, predicteds, color="gray", alpha=0.2)
            plt.title(f"{dataset} - Best Model: Actual vs Predicted RUL (Sorted)")
            plt.xlabel("Test Engine Instance (Sorted by Actual RUL)")
            plt.ylabel("Remaining Useful Life (Cycles)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "prediction_vs_actual.png"))
            plt.close()
    except Exception as e:
        print(f"Failed to plot prediction_vs_actual: {e}")

    # 3. Error Distribution (Histogram)
    try:
        if best_preds:
            plt.figure(figsize=(8, 6))
            errors = [p["predicted_rul"] - p["actual_rul"] for p in best_preds]
            sns.histplot(errors, bins=15, kde=True, color="purple")
            plt.axvline(0, color="black", linestyle="--")
            plt.title(f"{dataset} - Prediction Error Distribution")
            plt.xlabel("Error (Predicted - Actual)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "error_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Failed to plot error_distribution: {e}")

    # 4. Model Comparison Bar Chart
    try:
        if all_metrics:
            plt.figure(figsize=(8, 6))
            models = list(all_metrics.keys())
            rmses = [all_metrics[m]["rmse"] for m in models]
            sns.barplot(x=models, y=rmses, palette="Blues_d")
            plt.title(f"{dataset} - Model RMSE Comparison")
            plt.ylabel("RMSE (Cycles)")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "model_comparison.png"))
            plt.close()
    except Exception as e:
        print(f"Failed to plot model_comparison: {e}")


# ── Training Pipeline For Combined Datasets ───────────────────────────────────

def train_combined(args):
    dataset = "combined"
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", dataset)
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", dataset)
    os.makedirs(models_dir, exist_ok=True)

    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n[combined] Loading and combining all dataset files...")
    
    # Offsets for combining dataset IDs to maintain uniqueness
    offsets = {"FD001": 1000, "FD002": 2000, "FD003": 3000, "FD004": 4000}
    
    train_raw_list, test_raw_list, rul_df_list = [], [], []
    
    for dst, (tr_file, te_file, ru_file) in DATASET_FILES.items():
        tr, te, ru = load_dataset(
            os.path.join(data_dir, tr_file),
            os.path.join(data_dir, te_file),
            os.path.join(data_dir, ru_file)
        )
        offset = offsets[dst]
        tr["unit"] += offset
        te["unit"] += offset
        ru["unit"] += offset
        
        train_raw_list.append(tr)
        test_raw_list.append(te)
        rul_df_list.append(ru)

    train_raw = pd.concat(train_raw_list, ignore_index=True)
    test_raw = pd.concat(test_raw_list, ignore_index=True)
    rul_df = pd.concat(rul_df_list, ignore_index=True)

    # ── 2. Preprocessing
    print(f"[combined] Preprocessing (EMA smoothing enabled)...")
    train_df, scaler, feature_cols = preprocess_train(train_raw, rul_cap=args.rul_cap)
    test_df  = preprocess_test(test_raw, rul_df, scaler, feature_cols, rul_cap=args.rul_cap)
    save_preprocessing_artifacts(scaler, feature_cols, models_dir, rul_cap=args.rul_cap)

    # ── 3. Windowing
    print(f"[combined] Creating sequence windows (size={args.window_size})...")
    X, y_reg, y_reg_raw, y_clf = create_sequences(train_df, feature_cols, window_size=args.window_size)
    X_reg_test, y_reg_test, y_raw_test, X_clf_test, y_clf_test = create_sequences_test_last(
        test_df, feature_cols, window_size=args.window_size
    )

    assert y_reg.max() <= 1.01, "BUG: y_reg is not normalized to [0,1]"
    
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_tr, X_val       = X[idx[:split]],       X[idx[split:]]
    y_reg_tr, y_reg_val = y_reg[idx[:split]], y_reg[idx[split:]]
    y_clf_tr, y_clf_val = y_clf[idx[:split]], y_clf[idx[split:]]

    class_weights  = make_class_weights(y_clf)
    sample_weights = build_sample_weight(y_clf_tr, class_weights)

    input_shape = (args.window_size, len(feature_cols))

    # ── 4. Train Model
    m_name = "hybrid"
    print(f"\n[combined] === Training {m_name.upper()} model on combined datasets ===")
    builder = build_hybrid_model
    model = compile_model(builder(input_shape=input_shape), learning_rate=args.lr)
    
    history = model.fit(
        X_tr, [y_reg_tr, y_clf_tr],
        sample_weight=sample_weights,
        validation_data=(X_val, [y_reg_val, y_clf_val]),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=get_callbacks(patience=args.patience),
        verbose=1,
    )

    metrics = evaluate_model(model, X_reg_test, y_raw_test, X_clf_test, y_clf_test, args.rul_cap)
    all_metrics = {m_name: metrics}
    all_histories = [serialize_history(history, m_name)]
    
    print(f"[combined] {m_name} Evaluation -> RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, F1: {metrics['f1']}")

    best_model_path = os.path.join(models_dir, "best_model.keras")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    model.save(best_model_path)

    # Re-load best model to extract final deterministic predictions output
    final_model = tf.keras.models.load_model(best_model_path)
    best_preds = get_per_engine_predictions(
        final_model, test_df, feature_cols, args.window_size, args.rul_cap, m_name
    )

    # ── Strict Validation Checks ──
    # Check for constant predictions / model collapse
    pred_ruls = np.array([p["predicted_rul"] for p in best_preds])
    if np.std(pred_ruls) < 2.0:
        print("[WARNING] Strict Validation Failed: Model predictions are nearly constant (potential model collapse/leakage).")
        
    # Check per-dataset performance explicitly
    per_dataset_metrics = {}
    for dst, offset in offsets.items():
        dst_preds = [p for p in best_preds if offset < p["engine_id"] <= offset + 999]
        if not dst_preds: continue
        
        preds_arr = np.array([p["predicted_rul"] for p in dst_preds])
        actuals_arr = np.array([p["actual_rul"] for p in dst_preds])
        dst_rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr))
        dst_mae = mean_absolute_error(actuals_arr, preds_arr)
        per_dataset_metrics[dst] = {"rmse": round(float(dst_rmse), 3), "mae": round(float(dst_mae), 3)}
        print(f"[{dst}] Validated Performance: RMSE={dst_rmse:.3f}, MAE={dst_mae:.3f}")

    # ── 6. Save JSON Manifest Artifacts ──────────────────────────────────────
    with open(os.path.join(models_dir, "model_info.json"), "w") as f:
        json.dump({
            "best_model": m_name, 
            "metrics": metrics,
            "description": "Generalized Hybrid CNN-LSTM model trained on combined FD001-FD004 CMAPSS dataset"
        }, f, indent=2)

    with open(os.path.join(models_dir, "best_model_info.json"), "w") as f:
        # For legacy compatibility during transition
        json.dump({
            "best_model": m_name, 
            "metrics": metrics,
        }, f, indent=2)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    with open(os.path.join(models_dir, "per_dataset_metrics.json"), "w") as f:
        json.dump(per_dataset_metrics, f, indent=2)

    with open(os.path.join(models_dir, "training_history.json"), "w") as f:
        json.dump(all_histories, f, indent=2)

    with open(os.path.join(models_dir, "test_predictions.json"), "w") as f:
        json.dump(best_preds, f, indent=2)
        
    engine_data = export_engine_data(test_df, feature_cols)
    with open(os.path.join(models_dir, "engine_data.json"), "w") as f:
        json.dump(engine_data, f)

    meta = {
        "dataset": "combined",
        "type": "generalized_hybrid",
        "window_size": args.window_size, "rul_cap": args.rul_cap,
        "features": feature_cols, "trained_models": [m_name],
        "best_model": m_name
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── 7. Automatic Plot Generation ─────────────────────────────────────────
    generate_plots(dataset, all_histories, best_preds, all_metrics, plots_dir)
    print(f"[combined] Done! Artifacts populated into models/{dataset} and plots generated in plots/{dataset}.\n")


# ── Run Controller ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generalized Multi-Dataset CMAPSS Predictor Train Pipeline")
    p.add_argument("--epochs",       type=int,   default=30,   help="Max training epochs")
    p.add_argument("--batch_size",   type=int,   default=128,  help="Mini-batch size")
    p.add_argument("--window_size",  type=int,   default=50,   help="Sliding window length (was 30)")
    p.add_argument("--rul_cap",      type=int,   default=125,  help="Piecewise RUL cap")
    p.add_argument("--lr",           type=float, default=5e-4, help="Initial learning rate (was 1e-3)")
    p.add_argument("--patience",     type=int,   default=12,   help="EarlyStopping patience (was 5)")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_combined(args)
