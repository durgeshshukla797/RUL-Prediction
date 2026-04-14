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
from models_arch.lstm_only import build_lstm_model
from models_arch.cnn_only import build_cnn_model

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

def compile_model(model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={"rul": "mse", "health": "sparse_categorical_crossentropy"},
        loss_weights={"rul": 1.0, "health": 0.3},
        metrics={"rul": ["mae"], "health": ["accuracy"]},
    )
    return model

def get_callbacks(patience: int = 10):
    return [
        EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(3, patience // 3), min_lr=1e-6, verbose=1,
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


# ── Training Pipeline For A Single Dataset ────────────────────────────────────

def train_dataset(dataset: str, args):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", dataset)
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", dataset)
    os.makedirs(models_dir, exist_ok=True)

    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_file, test_file, rul_file = DATASET_FILES[dataset]
    print(f"\n[{dataset}] Loading dataset files...")
    
    # ── 1. Load Data 
    train_raw, test_raw, rul_df = load_dataset(
        os.path.join(data_dir, train_file),
        os.path.join(data_dir, test_file),
        os.path.join(data_dir, rul_file),
    )

    # ── 2. Preprocessing
    print(f"[{dataset}] Preprocessing (EMA smoothing enabled)...")
    train_df, scaler, feature_cols = preprocess_train(train_raw, rul_cap=args.rul_cap)
    test_df  = preprocess_test(test_raw, rul_df, scaler, feature_cols, rul_cap=args.rul_cap)
    save_preprocessing_artifacts(scaler, feature_cols, models_dir, rul_cap=args.rul_cap)

    # ── 3. Windowing
    print(f"[{dataset}] Creating sequence windows (size={args.window_size})...")
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

    # ── 4. Train Models
    model_builders = {
        "hybrid": build_hybrid_model,
        "lstm": build_lstm_model,
        "cnn": build_cnn_model,
    }

    request = ["hybrid", "lstm", "cnn"] if args.model == "all" else [args.model]
    
    all_metrics = {}
    all_histories = []
    keras_paths = {}

    for m_name in request:
        print(f"\n[{dataset}] === Training {m_name.upper()} model ===")
        builder = model_builders[m_name]
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
        all_metrics[m_name] = metrics
        all_histories.append(serialize_history(history, m_name))
        
        print(f"[{dataset}] {m_name} Evaluation -> RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, F1: {metrics['f1']}")

        # Temporary save all requested models so we can rank them safely
        temp_path = os.path.join(models_dir, f"{m_name}_temp.keras")
        model.save(temp_path)
        keras_paths[m_name] = temp_path

    # ── 5. Best Model Selection Workflow ──────────────────────────────────────
    # Sort strictly: RMSE (asc) -> MAE (asc) -> F1 (desc)
    ranked_models = sorted(all_metrics.keys(), key=lambda m: (
        all_metrics[m]["rmse"],
        all_metrics[m]["mae"],
        -all_metrics[m]["f1"]
    ))
    
    best_model_name = ranked_models[0]
    best_metrics = all_metrics[best_model_name]
    print(f"\n[{dataset}] * BEST MODEL IDENTIFIED: {best_model_name.upper()} *")
    print(f"[{dataset}] * METRICS: RMSE={best_metrics['rmse']} | MAE={best_metrics['mae']} | Accuracy={best_metrics['accuracy']} *")

    # Rename best model strictly into inference loader form
    best_model_path = os.path.join(models_dir, "best_model.keras")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    os.rename(keras_paths[best_model_name], best_model_path)

    # Cleanup losers
    for m_name, p in keras_paths.items():
        if m_name != best_model_name and os.path.exists(p):
            os.remove(p)
            
    # Re-load best model to extract final deterministic predictions output
    final_model = tf.keras.models.load_model(best_model_path)
    best_preds = get_per_engine_predictions(
        final_model, test_df, feature_cols, args.window_size, args.rul_cap, best_model_name
    )

    # ── 6. Save JSON Manifest Artifacts ──────────────────────────────────────
    with open(os.path.join(models_dir, "best_model_info.json"), "w") as f:
        json.dump({"best_model": best_model_name, "metrics": best_metrics}, f, indent=2)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(models_dir, "training_history.json"), "w") as f:
        json.dump(all_histories, f, indent=2)

    with open(os.path.join(models_dir, "test_predictions.json"), "w") as f:
        json.dump(best_preds, f, indent=2)
        
    engine_data = export_engine_data(test_df, feature_cols)
    with open(os.path.join(models_dir, "engine_data.json"), "w") as f:
        json.dump(engine_data, f)

    meta = {
        "dataset": dataset,
        "window_size": args.window_size, "rul_cap": args.rul_cap,
        "features": feature_cols, "trained_models": request,
        "best_model": best_model_name
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── 7. Automatic Plot Generation ─────────────────────────────────────────
    generate_plots(dataset, all_histories, best_preds, all_metrics, plots_dir)
    print(f"[{dataset}] Done! Artifacts populated into models/{dataset} and plots generated in plots/{dataset}.\n")


# ── Run Controller ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Dataset CMAPSS Predictor Train Pipeline")
    p.add_argument("--dataset",      default="all", help="all | FD001 | FD002...")
    p.add_argument("--model",        default="all", choices=["all", "hybrid", "lstm", "cnn"])
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--window_size",  type=int,   default=30)
    p.add_argument("--rul_cap",      type=int,   default=125)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    targets = list(DATASET_FILES.keys()) if args.dataset == "all" else [args.dataset.upper()]
    
    for dst in targets:
        if dst in DATASET_FILES:
            train_dataset(dst, args)
        else:
            print(f"Unknown dataset {dst}. Skipping.")
