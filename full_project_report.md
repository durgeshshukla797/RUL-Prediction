# Predictive Maintenance System for Turbofan Engines
## Complete Project Technical Report

---

## Table of Contents

1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [Problem Statement](#2-problem-statement)
3. [Dataset — NASA CMAPSS](#3-dataset--nasa-cmapss)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
5. [Sliding Window (Sequence Generation)](#5-sliding-window-sequence-generation)
6. [How Training & Test Data Are Used](#6-how-training--test-data-are-used)
7. [Model Architecture — Hybrid CNN-LSTM](#7-model-architecture--hybrid-cnn-lstm)
8. [Training Configuration & Hyperparameters](#8-training-configuration--hyperparameters)
9. [Loss Function & Optimization](#9-loss-function--optimization)
10. [Evaluation & Results](#10-evaluation--results)
11. [Backend — FastAPI REST API](#11-backend--fastapi-rest-api)
12. [Frontend — React Dashboard](#12-frontend--react-dashboard)
13. [Full Tech Stack](#13-full-tech-stack)
14. [Project File Structure](#14-project-file-structure)
15. [End-to-End Flow Summary](#15-end-to-end-flow-summary)

---

## 1. Project Overview & Motivation

### What is Predictive Maintenance?

Traditional aircraft engine maintenance follows one of two schedules:

- **Reactive maintenance** — fix the engine after it breaks (dangerous, unacceptable for aviation)
- **Preventive maintenance** — replace/service at fixed intervals regardless of actual condition (very expensive, often wasteful)

**Predictive maintenance** is the third, smarter approach: continuously monitor the engine's health using real sensor data and predict *exactly* when it will fail — so you can service it just in time, neither too early (wasting life) nor too late (risking failure).

### The Goal of this Project

Build an end-to-end AI system that:

1. Takes sensor readings from a jet engine (temperature, pressure, RPM, fuel flow, etc.)
2. Predicts the **Remaining Useful Life (RUL)** — how many flight cycles the engine can safely complete before failure
3. Classifies the engine's **health state** — Critical 🚨 / Warning ⚠️ / Healthy ✅
4. Serves predictions via a REST API with a real-time web dashboard

---

## 2. Problem Statement

Given a **time series of sensor measurements** from a turbofan engine, predict:

- **Regression target:** `RUL` — the number of remaining operational cycles before failure
- **Classification target:** Health class — `0=Critical (RUL ≤ 15)`, `1=Warning (15 < RUL ≤ 30)`, `2=Healthy (RUL > 30)`

This is a **supervised multi-task learning** problem — one model simultaneously solves both tasks.

**Why multi-task?** RUL regression and health classification are strongly correlated. Jointly training them forces the shared representation to capture degradation patterns that benefit both tasks, improving generalization.

---

## 3. Dataset — NASA CMAPSS

### What is CMAPSS?

**CMAPSS** = *Commercial Modular Aero-Propulsion System Simulation*

A turbofan engine degradation simulation dataset created by NASA's Ames Research Center. It simulates run-to-failure experiments on turbofan jet engines under various operating conditions and fault modes.

> **Real-world analogy:** Imagine attaching 21 medical sensors to an airplane engine, flying it thousands of times, and recording every sensor at each takeoff-to-landing cycle until the engine fails. CMAPSS is that exact experiment, simulated at high fidelity.

### The 4 Sub-Datasets

| Dataset | Train Engines | Test Engines | Operating Conditions | Fault Modes | Difficulty |
|---------|:------------:|:------------:|:-------------------:|:-----------:|:----------:|
| **FD001** | 100 | 100 | 1 | 1 (HPC degradation) | ✅ Easiest |
| **FD002** | 260 | 259 | 6 | 1 (HPC degradation) | ⚠️ Medium |
| **FD003** | 100 | 100 | 1 | 2 (HPC + Fan) | ⚠️ Medium |
| **FD004** | 249 | 248 | 6 | 2 (HPC + Fan) | ❌ Hardest |
| **Total** | **709** | **707** | — | — | — |

- **HPC** = High-Pressure Compressor
- **Operating Conditions** = combinations of altitude, throttle resolver angle, and Mach number
- **1 operating condition** = all flights are identical (sea-level cruise, same throttle)
- **6 operating conditions** = different flight profiles: some short-haul, some ultra-long-haul

### The 3 File Types Per Dataset

Each of the 4 datasets has exactly 3 files:

```
train_FD001.txt   — Engine degradation histories, from brand new → failure
test_FD001.txt    — Partial engine histories (cut off before failure)
RUL_FD001.txt     — Ground truth: how many cycles each test engine had left
```

### Raw Data Format

Each file is **whitespace-delimited** with **26 columns** and no header:

```
unit  time  c1       c2       c3     s1      s2      s3      ... s21
1     1     -0.0007  -0.0004  100.0  518.67  641.82  1589.70 ... 23.419
1     2      0.0019  -0.0003  100.0  518.67  642.15  1591.82 ... 23.424
...
1     192    0.0012  -0.0004  100.0  518.67  642.11  1589.81 ... 23.411  ← engine dies
2     1     ...
```

| Column | Name | Description |
|--------|------|-------------|
| 1 | `unit` | Engine ID (1-indexed) |
| 2 | `time` | Cycle number (flight count for this engine) |
| 3 | `c1` | Operational setting 1 — Altitude (sea-level deviation, ~0 for FD001) |
| 4 | `c2` | Operational setting 2 — Throttle resolver angle deviation |
| 5 | `c3` | Operational setting 3 — Altitude in 1000ft |
| 6–26 | `s1`–`s21` | 21 sensor measurements (see table below) |

### The 21 Sensors

| Sensor | Description | Unit | Informative? |
|--------|-------------|------|:---:|
| s1 | Total temperature at fan inlet | °R | ❌ Near-constant |
| s2 | Total temperature at LPC outlet | °R | ✅ Used |
| s3 | Total temperature at HPC outlet | °R | ✅ Used |
| s4 | Total temperature at LPT outlet | °R | ✅ Used |
| s5 | Pressure at fan inlet | psia | ❌ Near-constant |
| s6 | Total pressure at bypass-duct | psia | ✅ Used |
| s7 | Total pressure at HPC outlet | psia | ✅ Used |
| s8 | Physical fan speed | rpm | ✅ Used |
| s9 | Physical core speed | rpm | ✅ Used |
| s10 | Engine pressure ratio | — | ❌ Near-constant |
| s11 | Static pressure at HPC outlet | psia | ✅ Used |
| s12 | Ratio of fuel flow to Ps30 | pps/psi | ✅ Used |
| s13 | Corrected fan speed | rpm | ✅ Used |
| s14 | Corrected core speed | rpm | ✅ Used |
| s15 | Bypass ratio | — | ✅ Used |
| s16 | Burner fuel-air ratio | — | ❌ Near-constant |
| s17 | Bleed enthalpy | — | ✅ Used |
| s18 | Demanded fan speed | rpm | ❌ Near-constant |
| s19 | Demanded corrected fan speed | rpm | ❌ Near-constant |
| s20 | HPT coolant bleed | lbm/s | ✅ Used |
| s21 | LPT coolant bleed | lbm/s | ✅ Used |

**Dropped sensors:** `s1, s5, s10, s16, s18, s19` — near-zero variance across all engines, carry no degradation signal.

**Final feature set (18 features):** `c1, c2, c3, s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`

---

## 4. Data Preprocessing Pipeline

### Step 1 — RUL Label Computation

**For training data** (full engine lives known):

```
RUL_raw(t) = min(max_cycle_of_engine - current_cycle, 125)
RUL_norm   = RUL_raw / 125   ← normalized to [0, 1] for neural network target
```

**Why cap at 125?** This is the *piecewise-linear RUL* approach validated in academic literature. Engines with more than 125 cycles left are all equally healthy — the model doesn't need to distinguish "500 cycles left" from "200 cycles left". The cap prevents biasing the model toward predicting very high RUL values.

**For test data** (partial histories, end-of-life unknown):

```
RUL at last cycle is given by RUL_FD001.txt
RUL at earlier cycles = RUL_at_last_cycle + (max_cycle - current_cycle), capped at 125
```

**Example — Engine #1 (FD001), died at cycle 192:**

| Cycle | RUL Raw | RUL Normalized | Health Label |
|-------|:-------:|:--------------:|:------------:|
| 1 | 125 (capped) | 1.000 | Healthy |
| 68 | 124 | 0.992 | Healthy |
| 100 | 92 | 0.736 | Healthy |
| 163 | 29 | 0.232 | Warning |
| 178 | 14 | 0.112 | Critical |
| 192 | 0 | 0.000 | Critical |

### Step 2 — Health Label Assignment

Derived directly from `RUL_raw`:

```
RUL_raw  > 30  →  Class 2: Healthy   ✅
15 < RUL_raw ≤ 30  →  Class 1: Warning  ⚠️
RUL_raw ≤ 15  →  Class 0: Critical  🚨
```

### Step 3 — Drop Zero-Variance Sensors

```python
CONSTANT_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}
```

Additionally, any sensor whose standard deviation across the entire training set is below `1e-4` is automatically removed (this catches `c3` in FD001 which is stuck at a constant value).

### Step 4 — EMA Smoothing (Exponential Moving Average)

Sensor readings contain random noise from measurement error. EMA smoothing removes this jitter while preserving the real degradation trend:

```python
df[feature_cols] = df[feature_cols].ewm(alpha=0.4, adjust=False).mean()
```

- Applied **per engine** (not across engines, which would be data leakage)
- `alpha=0.4` → 40% weight on current reading, 60% on exponentially weighted history
- Effect: sharp noise spikes are smoothed; gradual degradation trends are preserved

**Analogy:** Like smoothing out a bumpy road — the direction of travel (trend) is preserved but the micro-bumps (noise) are removed.

### Step 5 — MinMax Normalization

```python
scaler = MinMaxScaler()
scaler.fit(train_df[feature_cols].values)    # ← FIT ONLY on training data
scaler.transform(train_df[feature_cols])     # ← Apply to train
scaler.transform(test_df[feature_cols])      # ← Apply same scaler to test
```

All 18 features scaled to `[0, 1]`.

> **CRITICAL: The scaler is fitted exclusively on training data and then applied to test data using the same parameters. Fitting on test data would mean the model implicitly learns test sensor ranges — a form of data leakage that inflates evaluation scores.**

The scaler is saved to `models/combined/scaler.pkl` at training time and loaded at inference time to ensure identical normalization for new data.

### Step 6 — Dataset Merging with ID Offsets

All 4 datasets are combined into one training set with **engine ID offsets** to prevent collision:

```
FD001 engines → IDs 1001–1100   (offset +1000)
FD002 engines → IDs 2001–2260   (offset +2000)
FD003 engines → IDs 3001–3100   (offset +3000)
FD004 engines → IDs 4001–4249   (offset +4000)
```

This is critical: without offsets, Engine #1 from FD001 and Engine #1 from FD002 would be treated as the same engine, corrupting RUL calculations.

---

## 5. Sliding Window (Sequence Generation)

### Why Windows?

The model needs temporal context — the pattern of *how sensors change over time* is what reveals degradation, not just a single snapshot. A single row like "temperature = 642°F" is meaningless alone. But "temperature was 640, 641, 643, 644, 645, 647... (rising over 50 cycles)" tells the model the engine is degrading.

### How It Works

For each engine, a **sliding window of 50 consecutive cycles** is extracted:

```
Engine #1 has 192 cycles. window_size = 50. stride = 1.

Window 001: [cycles  1→50 ]  → target RUL = 125 (capped), Health = Healthy
Window 002: [cycles  2→51 ]  → target RUL = 124,          Health = Healthy
Window 003: [cycles  3→52 ]  → target RUL = 123,          Health = Healthy
...
Window 143: [cycles 143→192] → target RUL = 0,            Health = Critical
```

Each window becomes a 3D tensor: **shape (50, 18)** — 50 time steps × 18 sensor features.

### Training Data Shape

All windows from all 709 engines are stacked:

```
X       : (N_windows, 50, 18)  — input to model
y_reg   : (N_windows,)          — normalized RUL target [0, 1]
y_clf   : (N_windows,)          — health class target [0, 1, 2]
```

With ~100 windows per engine × 709 engines ≈ **~70,000–100,000 training windows**.

### Test Data — Last Window Only

For test engines, only the **final window** (last 50 cycles) matters for RUL prediction. This simulates the real-world scenario: you have the most recent sensor readings and must predict how much life remains.

```python
# windowing.py L103
if start == n - window_size:   # ONLY the last window per engine
    X_reg_list.append(x)
```

### Padding for Short Engines

If a test engine has fewer than 50 cycles of history, the sequence is **front-padded** by repeating the earliest known reading:

```
Engine with 31 cycles, window=50:
[row1, row1, ..., row1, cycle1, cycle2, ..., cycle31]
 ←  19 padding rows  →  ←       31 real rows      →
Final shape: (1, 50, 18) — still valid input to model
```

---

## 6. How Training & Test Data Are Used

### Training Data Role

Training data contains engines that ran **to complete failure**. Because we know when each engine died, the preprocessing can compute exact RUL for every single cycle. The model learns from these complete degradation histories.

After preprocessing → windowing → the training windows are split **80/20**:

```python
idx   = np.random.permutation(len(X))   # random shuffle of all windows
split = int(0.8 * len(X))
X_tr,  X_val   = X[idx[:split]], X[idx[split:]]
```

| Split | % | Used For |
|-------|---|----------|
| **Training set** | 80% | Model computes loss on these, gradients flow, weights update |
| **Validation set** | 20% | Model **never trains** on these; used to detect overfitting and trigger early stopping |

### Test Data Role

Test data contains engines **cut off mid-life** — we see the engine up to some point, don't know when it will fail. The model must predict remaining cycles from the most recent 50-cycle window.

After prediction, results are compared against `RUL_FD001.txt` (the answer key) to compute RMSE/MAE. This is the **final honest evaluation** — the model never saw test data during training.

### The Complete Data Lifecycle

```
train_FD001→FD004.txt    test_FD001→FD004.txt    RUL_FD001→FD004.txt
      │                         │                        │
      ▼                         ▼                        ▼
  Add RUL labels           Add RUL labels          (answer key, not
  Add health class         Add health class          touched until eval)
  Drop constants           Apply SAME scaler
  EMA smooth               (no fitting here)
  Fit+Apply scaler
      │                         │
      ▼                         ▼
  Sliding windows          Last window per engine
  (all positions)          (1 per engine only)
      │                         │
      ▼                         ▼
  80% train / 20% val      Feed into trained model
      │                         │
  CNN-LSTM learns           Predict RUL + Health
      │                         │
  Weights updated           Compare vs answer key
                                │
                            RMSE / MAE / F1
```

### Class Imbalance Handling

Most cycles in training are "Healthy" (engine starts new, spends most time healthy). Critical cycles are rare. Without correction, the model would just predict "Healthy" always:

```python
class_weights = compute_class_weight("balanced", classes=[0,1,2], y=y_clf)
# Typical result: {0: 4.2, 1: 3.1, 2: 0.7}
# Critical windows weighted ~6x more than Healthy ones
```

This forces the model to pay 6× more attention to critical degradation patterns.

---

## 7. Model Architecture — Hybrid CNN-LSTM

### Why Hybrid CNN-LSTM?

- **CNN (Convolutional Neural Network)** — excels at extracting local patterns: "what changed between cycle 45 and cycle 50?" Works like a sliding magnifying glass across the time dimension.
- **LSTM (Long Short-Term Memory)** — excels at long-range temporal dependencies: "the engine's temperature has been gradually rising since cycle 1." Remembers patterns across many timesteps.
- **Combining both** — CNN first extracts local sensor correlations, LSTM then models how those patterns evolve over the 50-cycle window.

### Architecture (v2 — Improved)

```
Input: (50 timesteps × 18 sensor features)
            │
            ▼
┌─────────────────────────────────────────────────────┐
│                  CNN BLOCK                          │
│  Conv1D(64 filters, kernel=5, relu, padding=same)   │
│  BatchNormalization                                  │
│  Conv1D(128 filters, kernel=3, relu, padding=same)  │
│  BatchNormalization                                  │
│  MaxPool1D(pool=2)  →  s  │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│               BILSTM BLOCK 1                        │
│  Bidirectional LSTM(96 units, return_sequences=True)│
│  Dropout(0.35)                                      │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│               BILSTM BLOCK 2                        │
│  Bidirectional LSTM(64 units, return_sequences=True)│
│  Dropout(0.35)                                      │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│           TEMPORAL SELF-ATTENTION                   │
│  Dense(1, tanh) → attention scores per timestep     │
│  Softmax(axis=1) → attention weights (sum=1)        │
│  Multiply(x, weights) → weighted timestep features  │
│  GlobalAveragePool → collapse 25 steps → 1 vector   │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│             SHARED BOTTLENECK                       │
│  Dense(64, relu, L2=1e-4 regularization)            │
│  Dropout(0.35)                                      │
└─────────────────────────────────────────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
Dense(1, linear)    Dense(3, softmax)
"rul" output        "health" output
(RUL ∈ [0,1])       (P(Critical), P(Warning), P(Healthy))
```

### Layer-by-Layer Explanation

| Layer | Output Shape | Purpose |
|-------|:------------:|---------|
| Input | (50, 18) | 50 cycles × 18 sensor features |
| Conv1D(64, k=5) | (50, 64) | Detect local 5-cycle sensor patterns |
| BatchNorm | (50, 64) | Stabilize activations, allow higher learning rate |
| Conv1D(128, k=3) | (50, 128) | Deeper local features from the 64-feature maps |
| BatchNorm | (50, 128) | Stabilize again |
| MaxPool1D(2) | (25, 128) | Halve temporal dimension, force abstraction |
| BiLSTM(96) | (25, 192) | Read sequence forward AND backward (96×2=192) |
| Dropout(0.35) | (25, 192) | Random neuron drop to prevent overfitting |
| BiLSTM(64) | (25, 128) | Deeper bidirectional temporal modeling |
| Dropout(0.35) | (25, 128) | Regularization |
| Attention Dense(1) | (25, 1) | Score each of 25 timesteps for importance |
| Softmax | (25, 1) | Normalize scores → probability weights |
| Multiply | (25, 128) | Weight each timestep's features by its importance |
| GlobalAvgPool | (128,) | Aggregate 25 weighted timesteps → one vector |
| Dense(64, relu) | (64,) | Shared representation for both heads |
| Dropout(0.35) | (64,) | Final regularization |
| **RUL head**: Dense(1, linear) | **(1,)** | **Predict normalized RUL ∈ [0, 1]** |
| **Health head**: Dense(3, softmax) | **(3,)** | **Predict class probabilities** |

### Why Bidirectional LSTM?

Standard LSTM reads the sequence left-to-right (past → present). Bidirectional reads it both ways simultaneously. For degradation patterns, this is valuable because degradation at timestep t=30 provides context for understanding what happened at t=25. Reading both directions captures the fullest temporal context.

### Why Attention?

Not all 25 timesteps are equally important. The last few cycles before a failure (where degradation accelerates) are far more informative than cycles 1–5. The attention layer learns to weight the most diagnostic timesteps more heavily, dynamically per input.

### Total Trainable Parameters: ~1.2 million

---

## 8. Training Configuration & Hyperparameters

| Hyperparameter | Value | Rationale |
|---------------|:-----:|-----------|
| `window_size` | 50 | Captures 50-cycle degradation context; validated against 30 |
| `rul_cap` | 125 | Standard piecewise-linear RUL approach from literature |
| `batch_size` | 128 | Balances GPU memory and gradient quality |
| `epochs` (max) | 30 | Hard upper limit; EarlyStopping typically stops before this |
| `learning_rate` | 5×10⁻⁴ | Conservative start; ReduceLROnPlateau halves when plateaued |
| `patience` | 12 | EarlyStopping waits 12 epochs before stopping |
| `LR_patience` | 4 | ReduceLROnPlateau halves LR after 4 plateau epochs |
| `min_lr` | 1×10⁻⁶ | Floor for learning rate |
| `dropout` | 0.35 | Applied after both BiLSTMs and the shared Dense |
| `L2 penalty` | 1×10⁻⁴ | Applied to shared Dense layer weights |
| `gradient_clip` | 1.0 (clipnorm) | Prevents exploding gradients during backprop |
| `EMA alpha` | 0.4 (train), 0.3 (test) | Slightly more smoothing on training; moderate on test |
| `val_split` | 20% | Held-out from training for overfitting detection |
| `optimizer` | Adam | Adaptive learning rate per parameter |
| `seed` | 42 | Reproducibility |

### Callbacks

```
EarlyStopping:
  - Monitor: val_rul_mae (validation Mean Absolute Error on RUL)
  - Patience: 12 epochs
  - Action: Stop training + restore best weights seen so far

ReduceLROnPlateau:
  - Monitor: val_rul_mae
  - Patience: 4 epochs
  - Factor: 0.5 (halve the learning rate)
  - Min LR: 1e-6
```

---

## 9. Loss Function & Optimization

### Dual Loss (Multi-Task Learning)

```
Total Loss = 1.5 × MSE(predicted_RUL, true_RUL) + 0.3 × CrossEntropy(predicted_health, true_health)
```

| Component | Formula | Weight | Role |
|-----------|---------|:------:|------|
| RUL Loss | Mean Squared Error | **1.5×** | Primary objective — precise remaining life prediction |
| Health Loss | Sparse Categorical Cross-Entropy | **0.3×** | Secondary — health classification |

- RUL is weighted 5× more than health because accurate RUL prediction is the primary engineering goal
- Health classification acts as an auxiliary task that improves the model's latent representation without dominating training

### Optimizer: Adam with Gradient Clipping

```python
optimizer = Adam(learning_rate=5e-4, clipnorm=1.0)
```

- **Adam** (Adaptive Moment Estimation): adapts the learning rate per parameter based on first and second moment estimates of gradients → faster convergence than vanilla SGD
- **clipnorm=1.0**: if the global gradient norm exceeds 1.0, all gradients are scaled down proportionally → prevents training instability

### Inference Output Conversion

```
predicted_rul_cycles = model_output_rul × 125   (denormalize)
predicted_health     = argmax(model_output_health_probabilities)
```

---

## 10. Evaluation & Results

### Metrics Used

| Metric | Formula | Applied To |
|--------|---------|-----------|
| **RMSE** | √(mean((pred - actual)²)) | RUL regression (cycles) |
| **MAE** | mean(|pred - actual|) | RUL regression (cycles) |
| **Accuracy** | correct / total | Health classification |
| **F1 Score** | Weighted harmonic mean of precision/recall | Health classification |

### Global Model Performance (combined dataset)

| Metric | Value |
|--------|:-----:|
| RMSE | **18.596 cycles** |
| MAE | **13.683 cycles** |
| Accuracy | **96.42%** |
| F1 Score | **0.9691** |

### Per-Dataset Performance

| Dataset | RMSE (cycles) | MAE (cycles) | Difficulty |
|---------|:-------------:|:------------:|-----------|
| **FD001** | **14.649** | **10.419** | ✅ Easiest (1 condition, 1 fault) |
| **FD002** | 19.027 | 14.703 | ⚠️ Medium (6 conditions, 1 fault) |
| **FD003** | 15.641 | 10.828 | ⚠️ Medium (1 condition, 2 faults) |
| **FD004** | 20.631 | 15.178 | ❌ Hardest (6 conditions, 2 faults) |

**Interpretation:** The pattern is exactly as expected — more operating conditions and more fault modes increase prediction difficulty. FD004 (6 conditions, 2 faults) is hardest at 20.6 cycles RMSE; FD001 (1 condition, 1 fault) is easiest at 14.6 cycles RMSE.

An RMSE of ~15–20 cycles means on average, the model's prediction is off by about 15–20 flight cycles. For engines with 100–300 cycle lifespans, this is roughly a **5–15% error**, which is considered very good performance on the CMAPSS benchmark.

### Validation: Collapse Detection

```python
pred_ruls = np.array([p["predicted_rul"] for p in best_preds])
if np.std(pred_ruls) < 2.0:
    print("[WARNING] Model predictions are nearly constant — potential collapse.")
```

A well-trained model should produce a wide spread of RUL predictions matching the diverse test engines. If std < 2.0, the model has collapsed to predicting a constant value.

---

## 11. Backend — FastAPI REST API

### Technology

- **Framework:** FastAPI v0.110+ (Python)
- **Server:** Uvicorn (ASGI, async HTTP)
- **Port:** 8000
- **Run command:** `uvicorn api.app:app --reload --port 8000`

### Startup Behavior

On server start, the combined model is **eagerly loaded into memory**:

```python
load_combined_state()   # called once at module load
```

Loads:
- `models/combined/best_model.keras` — the trained Keras model
- `models/combined/scaler.pkl` — the fitted MinMaxScaler
- `models/combined/feature_cols.json` — list of 18 features
- `models/combined/metadata.json` — window_size, rul_cap, etc.
- `models/combined/metrics.json` — performance numbers
- `models/combined/test_predictions.json` — pre-computed test engine results
- `models/combined/engine_data.json` — all test engine sensor histories

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns `{status: "ok"}` |
| `GET` | `/engines` | Lists all available engine IDs and count |
| `GET` | `/engine/{id}` | Full sensor history + pre-computed RUL for one engine |
| `GET` | `/engine/{id}/predict/{cycle}` | Real-time prediction at a specific cycle |
| `POST` | `/predict` | Body: `{engine_id, cycle}` → prediction |
| `POST` | `/upload` | Upload a CSV file → batch predictions for all engines |
| `GET` | `/metrics` | Model performance metrics (RMSE, MAE, F1) |
| `GET` | `/history` | Training loss/val_loss curves |
| `GET` | `/predictions` | All test engine pre-computed predictions |
| `GET` | `/best_model_info` | Best model name and metrics |
| `GET` | `/per_dataset_metrics` | Per-dataset RMSE/MAE breakdown |
| `GET` | `/plots/{filename}` | Static file server for training plots |

### Inference Flow (per request)

```
Request: GET /engine/1005/predict/30

1. Load engine_data["1005"] from state (in-memory, instant)
2. Collect all cycles ≤ 30 from the engine's history
3. Build a DataFrame: [(unit, time, feature1, ..., feature18)]
4. Apply create_inference_sequence() → pad if < 50 cycles → shape (1, 50, 18)
5. model.predict(X) → [rul_norm (1,1), health_proba (1,3)]
6. rul_cycles = rul_norm × 125
7. health_class = argmax(health_proba)
8. Return JSON: {engine_id, cycle, predicted_rul, health_label, confidence, ...}
```

### CSV Upload Flow

```
POST /upload (multipart CSV file)

1. Read and decode file content
2. Parse as whitespace-delimited DataFrame (same 26-column format)
3. Apply EMA smoothing (alpha=0.3)
4. Apply pre-fitted scaler (same one used during training)
5. For each engine in the file:
   a. Extract last 50 cycles (or pad if fewer)
   b. Run model.predict()
   c. Collect result
6. Return: {n_engines, predictions: [{engine_id, predicted_rul, health_label, ...}]}
```

---

## 12. Frontend — React Dashboard

### Technology

- **Framework:** React 18 with Vite build tool
- **Port:** 5173 (development server)
- **Run command:** `npm run dev` (from `frontend/`)
- **API communication:** Custom `fetch` wrapper in `src/api/client.js`
- **Styling:** Vanilla CSS (dark theme, glassmorphism, gradient accents)
- **Charts:** Recharts (line charts, scatter plots)

### Pages / Components

#### Dashboard (`App.jsx` — main page)
- Dataset selector (FD001–FD004, Combined)
- Model performance metrics cards (RMSE, MAE, F1, Accuracy)
- Per-dataset metrics comparison table
- Engine selector dropdown
- Per-engine: sensor trend line charts (all 18 features over time)
- Actual vs Predicted RUL scatter plot
- Training loss curves (train vs validation)
- Pre-computed test engine predictions table
- Health status distribution summary

#### Simulation Panel
- Select any test engine
- Play / Pause / Reset controls
- Speed selector (0.5×, 1×, 2×, 5×)
- Cycle-by-cycle animation — each tick calls `GET /engine/{id}/predict/{cycle}`
- Live RUL counter (updates each tick)
- Live health status badge (color-coded: green/yellow/red)
- RUL trend sparkline (mini chart)

#### Upload Page (`UploadPage.jsx`)
- Drag-and-drop or click-to-upload CSV file
- Sample CSV download button (pre-built example with 5 engines)
- On upload → `POST /upload` → batch predictions table
- After upload: integrated simulation panel
  - Pick any engine from the uploaded CSV
  - Client-side cycle-by-cycle simulation using uploaded data

---

## 13. Full Tech Stack

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| Deep Learning | TensorFlow / Keras | ≥ 2.15.0 |
| Data Processing | Pandas | ≥ 2.0.0 |
| Numerical | NumPy | ≥ 1.26.0 |
| ML Utilities | Scikit-learn | ≥ 1.4.0 |
| REST API | FastAPI | ≥ 0.110.0 |
| ASGI Server | Uvicorn | ≥ 0.29.0 |
| File Upload | python-multipart | ≥ 0.0.9 |
| Model Persistence | Keras `.keras` format + Pickle | — |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | React | 18 |
| Build Tool | Vite | Latest |
| Charts | Recharts | Latest |
| HTTP Client | Native Fetch API | — |
| Styling | Vanilla CSS (custom design system) | — |

### Training Outputs (artifacts saved to disk)

| File | Location | Contents |
|------|----------|----------|
| `best_model.keras` | `models/combined/` | Trained Keras model weights |
| `scaler.pkl` | `models/combined/` | Fitted MinMaxScaler |
| `feature_cols.json` | `models/combined/` | List of 18 feature names |
| `metadata.json` | `models/combined/` | window_size, rul_cap, dataset info |
| `metrics.json` | `models/combined/` | RMSE, MAE, F1, Accuracy |
| `per_dataset_metrics.json` | `models/combined/` | Per-FD RMSE/MAE |
| `training_history.json` | `models/combined/` | Loss curves per epoch |
| `test_predictions.json` | `models/combined/` | Per-engine actual vs predicted |
| `engine_data.json` | `models/combined/` | All test engines sensor histories |
| `loss_curve.png` | `plots/combined/` | Training vs validation loss |
| `prediction_vs_actual.png` | `plots/combined/` | Predicted vs actual RUL |
| `error_distribution.png` | `plots/combined/` | Histogram of prediction errors |
| `model_comparison.png` | `plots/combined/` | RMSE comparison chart |

---

## 14. Project File Structure

```
nasa/                                ← Project root
│
├── train.py                         ← Main training script (end-to-end pipeline)
├── requirements.txt                 ← Python dependencies
├── sample_engine_data.csv           ← Sample data for testing the upload feature
│
├── train_FD001.txt                  ← Training data: 100 engines, 1 condition, 1 fault
├── train_FD002.txt                  ← Training data: 260 engines, 6 conditions, 1 fault
├── train_FD003.txt                  ← Training data: 100 engines, 1 condition, 2 faults
├── train_FD004.txt                  ← Training data: 249 engines, 6 conditions, 2 faults
├── test_FD001.txt                   ← Test data: 100 partial engine histories
├── test_FD002.txt                   ← Test data: 259 partial engine histories
├── test_FD003.txt                   ← Test data: 100 partial engine histories
├── test_FD004.txt                   ← Test data: 248 partial engine histories
├── RUL_FD001.txt                    ← Ground truth RUL for 100 FD001 test engines
├── RUL_FD002.txt                    ← Ground truth RUL for 259 FD002 test engines
├── RUL_FD003.txt                    ← Ground truth RUL for 100 FD003 test engines
├── RUL_FD004.txt                    ← Ground truth RUL for 248 FD004 test engines
│
├── models_arch/
│   └── cnn_lstm.py                  ← HybridCNNLSTM_v2 model definition
│
├── utils/
│   ├── preprocessing.py             ← RUL computation, EMA, normalization
│   └── windowing.py                 ← Sliding window generation + inference helper
│
├── api/
│   └── app.py                       ← FastAPI backend (all endpoints)
│
├── models/
│   └── combined/
│       ├── best_model.keras         ← Trained model (saved after training)
│       ├── scaler.pkl               ← Fitted MinMaxScaler
│       ├── feature_cols.json        ← Feature list
│       ├── metadata.json            ← window_size, rul_cap
│       ├── metrics.json             ← Evaluation results
│       ├── per_dataset_metrics.json ← Per-FD results
│       ├── training_history.json    ← Loss curves
│       ├── test_predictions.json    ← Pre-computed predictions
│       └── engine_data.json         ← Test engine sensor data
│
├── plots/
│   └── combined/
│       ├── loss_curve.png
│       ├── prediction_vs_actual.png
│       ├── error_distribution.png
│       └── model_comparison.png
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx                 ← Entry point
        ├── App.jsx                  ← Main dashboard + routing
        ├── index.css                ← Global dark theme CSS
        ├── api/
        │   └── client.js            ← API helper functions
        └── components/
            ├── UploadPage.jsx       ← CSV upload + simulation panel
            └── [other components]  ← Charts, metric cards, engine viewer
```

---

## 15. End-to-End Flow Summary

### Training Phase

```
1. python train.py --epochs 30 --window_size 50

2. Load & merge FD001–FD004 train files (709 engines total)
3. Compute RUL for every row (backwards from failure cycle)
4. Assign health labels (0/1/2 from RUL thresholds)
5. Drop 6 zero-variance sensors → 18 features remain
6. EMA smooth (alpha=0.4) per engine
7. Fit MinMaxScaler on combined training features
8. Apply scaler → all features in [0,1]
9. Generate sliding windows: (N, 50, 18) tensors
10. 80/20 random split → training + validation sets
11. Compute class weights (balance Critical/Warning/Healthy)
12. Build CNN-LSTM model (~1.2M parameters)
13. train(X_train, y_reg_train, y_clf_train):
    - Batch size: 128
    - Max 30 epochs
    - Loss = 1.5×MSE_RUL + 0.3×CrossEntropy_Health
    - Adam(lr=5e-4, clipnorm=1.0)
    - EarlyStopping on val_rul_mae (patience=12)
    - ReduceLROnPlateau on val_rul_mae (patience=4, factor=0.5)
14. Evaluate on test engines (last window per engine)
15. Compare RUL predictions vs RUL_FD001-4.txt answer keys
16. Save model, scaler, metrics, plots to disk
```

### Inference Phase (API)

```
1. uvicorn api.app:app --reload --port 8000
   → Loads model + scaler + engine data into memory

2. User opens dashboard → React fetch to /metrics, /predictions, /engines

3. User selects engine #1005, plays simulation:
   → Every tick: GET /engine/1005/predict/{current_cycle}
   → Backend: build (1,50,18) window → model.predict → return RUL+health
   → Frontend: update live RUL number + health badge + sparkline

4. User uploads custom CSV:
   → POST /upload → parse → EMA smooth → apply saved scaler
   → For each engine in CSV: last 50 cycles → predict → return table
   → Frontend shows results + live simulation panel
```

### Key Design Principles

1. **No data leakage** — scaler fitted only on train, never on test
2. **Multi-task learning** — joint RUL + health training improves both
3. **Attention mechanism** — model focuses on diagnostically relevant timesteps
4. **Piecewise-linear RUL cap** — prevents bias toward high RUL values
5. **Generalized model** — trained on all 4 datasets simultaneously for robustness
6. **Class balance** — weighted loss prevents "always predict Healthy" collapse
7. **Early stopping** — automatically prevents overfitting

---

*Report generated from source code: `train.py`, `utils/preprocessing.py`, `utils/windowing.py`, `models_arch/cnn_lstm.py`, `api/app.py`, `models/combined/metrics.json`, `models/combined/per_dataset_metrics.json`*
