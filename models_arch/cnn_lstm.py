"""
models_arch/cnn_lstm.py
-----------------------
Hybrid CNN-LSTM model with dual output heads.

Architecture
------------
Input → Conv1D(32, k=5) → Conv1D(64, k=3) → MaxPool1D(3)
      → LSTM(50, ret_seq=True) → Dropout(0.2) → LSTM(50) → Dropout(0.2)
      → [Dense(1, linear)] regression head   (RUL)
      → [Dense(3, softmax)] classification head (health: 0=Critical,1=Warning,2=Healthy)
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dropout, Dense, BatchNormalization, Bidirectional
)


def build_hybrid_model(
    input_shape: tuple[int, int],
    cnn_filters: tuple[int, int] = (32, 64),
    kernel_sizes: tuple[int, int] = (5, 3),
    pool_size: int = 3,
    lstm_units: tuple[int, int] = (50, 50),
    dropout: float = 0.2,
    n_classes: int = 3,
) -> Model:
    """
    Build a Hybrid CNN-LSTM model with shared backbone and dual output heads.

    Parameters
    ----------
    input_shape : (window_size, n_features)
    cnn_filters : number of filters for each Conv1D layer
    kernel_sizes: kernel sizes for each Conv1D layer
    pool_size   : MaxPooling1D pool size
    lstm_units  : hidden units for each LSTM layer
    dropout     : dropout rate after each LSTM block
    n_classes   : number of health classes (default 3)

    Returns
    -------
    model : compiled Keras Model with two outputs ['rul', 'health']
    """
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

    # ── LSTM temporal modelling ────────────────────────────────────────────
    x = Bidirectional(LSTM(lstm_units[0], activation="tanh", return_sequences=True), name="bilstm1")(x)
    x = Dropout(dropout, name="drop1")(x)
    x = Bidirectional(LSTM(lstm_units[1], activation="tanh", return_sequences=False), name="bilstm2")(x)
    x = Dropout(dropout, name="drop2")(x)

    # ── Dual heads ─────────────────────────────────────────────────────────
    rul_output = Dense(1, activation="linear", name="rul")(x)
    health_output = Dense(n_classes, activation="softmax", name="health")(x)

    model = Model(inputs=inputs, outputs=[rul_output, health_output], name="HybridCNNLSTM")
    return model
