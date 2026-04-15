"""
models_arch/cnn_lstm.py
-----------------------
Hybrid CNN-LSTM model with dual output heads.

Architecture (v2 - improved)
-----------------------------
Input
  -> Conv1D(64, k=5) + BatchNorm
  -> Conv1D(128, k=3) + BatchNorm
  -> MaxPool1D(2)
  -> BiLSTM(96, return_sequences=True) + Dropout(0.35)
  -> BiLSTM(64, return_sequences=True)
  -> Attention pooling
  -> Dense(64, relu, L2) + Dropout(0.35)
  -> [Dense(1, linear)]     RUL regression head
  -> [Dense(3, softmax)]    Health classification head

Key improvements vs v1
-----------------------
* Larger CNN filters (32->64, 64->128) — richer local feature extraction
* Smaller pool_size (3->2) — avoids over-compressing short windows
* BiLSTM units increased (50->96/64) — more temporal capacity
* Dropout raised (0.2->0.35) — stronger regularization against overfitting
* Dense bottleneck (64 units) with L2 regularization added before heads
* Attention-style self-pooling layer — focus on degradation-relevant timesteps
* Returns sequences from first BiLSTM so second can attend over all steps
"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dropout, Dense,
    BatchNormalization, Bidirectional, GlobalAveragePooling1D,
    Multiply, Softmax, Reshape, Lambda,
)
from tensorflow.keras import regularizers


def build_hybrid_model(
    input_shape: tuple,
    cnn_filters: tuple = (64, 128),
    kernel_sizes: tuple = (5, 3),
    pool_size: int = 2,
    lstm_units: tuple = (96, 64),
    dropout: float = 0.35,
    dense_units: int = 64,
    l2: float = 1e-4,
    n_classes: int = 3,
) -> Model:
    """
    Build improved Hybrid CNN-LSTM with attention and regularization.

    Parameters
    ----------
    input_shape : (window_size, n_features)
    cnn_filters : filters for Conv1D layers
    kernel_sizes: kernel widths for Conv1D layers
    pool_size   : MaxPooling1D pool size
    lstm_units  : BiLSTM hidden units per layer
    dropout     : dropout rate (applied after each BiLSTM and Dense)
    dense_units : bottleneck dense layer size
    l2          : L2 weight decay for Dense layers
    n_classes   : health classification classes

    Returns
    -------
    model : compiled Keras Model with two outputs ['rul', 'health']
    """
    reg = regularizers.l2(l2)
    inputs = Input(shape=input_shape, name="sensor_sequence")

    # ── CNN feature extractor ──────────────────────────────────────────────
    x = Conv1D(
        filters=cnn_filters[0], kernel_size=kernel_sizes[0],
        activation="relu", padding="same", name="conv1"
    )(inputs)
    x = BatchNormalization(name="bn1")(x)

    x = Conv1D(
        filters=cnn_filters[1], kernel_size=kernel_sizes[1],
        activation="relu", padding="same", name="conv2"
    )(x)
    x = BatchNormalization(name="bn2")(x)
    x = MaxPooling1D(pool_size=pool_size, name="maxpool")(x)

    # ── BiLSTM temporal modelling ──────────────────────────────────────────
    x = Bidirectional(
        LSTM(lstm_units[0], activation="tanh", return_sequences=True),
        name="bilstm1"
    )(x)
    x = Dropout(dropout, name="drop1")(x)

    x = Bidirectional(
        LSTM(lstm_units[1], activation="tanh", return_sequences=True),
        name="bilstm2"
    )(x)
    x = Dropout(dropout, name="drop2")(x)

    # ── Temporal self-attention pooling ────────────────────────────────────
    # Learn which timesteps matter most for degradation
    attn_scores = Dense(1, activation="tanh", name="attn_score")(x)   # (B, T, 1)
    attn_weights = Softmax(axis=1, name="attn_weights")(attn_scores)  # (B, T, 1)
    x = Multiply(name="attn_context")([x, attn_weights])              # weighted
    x = GlobalAveragePooling1D(name="attn_pool")(x)                   # (B, F)

    # ── Shared bottleneck Dense ────────────────────────────────────────────
    x = Dense(dense_units, activation="relu",
               kernel_regularizer=reg, name="dense_shared")(x)
    x = Dropout(dropout, name="drop3")(x)

    # ── Dual heads ─────────────────────────────────────────────────────────
    rul_output    = Dense(1,         activation="linear",  name="rul")(x)
    health_output = Dense(n_classes, activation="softmax", name="health")(x)

    model = Model(
        inputs=inputs,
        outputs=[rul_output, health_output],
        name="HybridCNNLSTM_v2"
    )
    return model
