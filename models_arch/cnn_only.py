"""
models_arch/cnn_only.py
-----------------------
CNN-only model with dual output heads (for model comparison).
Uses Global Average Pooling to collapse the temporal dimension.
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
)


def build_cnn_model(
    input_shape: tuple[int, int],
    cnn_filters: tuple[int, int, int] = (32, 64, 128),
    kernel_sizes: tuple[int, int, int] = (7, 5, 3),
    dropout: float = 0.3,
    n_classes: int = 3,
) -> Model:
    """
    CNN-only model with three Conv1D layers, MaxPool, and GlobalAvgPool heads.

    Parameters
    ----------
    input_shape : (window_size, n_features)
    cnn_filters : filters per Conv1D layer
    kernel_sizes: kernel sizes per Conv1D layer
    dropout     : dropout rate before output heads
    n_classes   : number of health classes

    Returns
    -------
    model : Keras Model with two outputs ['rul', 'health']
    """
    inputs = Input(shape=input_shape, name="sensor_sequence")

    x = Conv1D(cnn_filters[0], kernel_sizes[0], activation="relu",
               padding="same", name="conv1")(inputs)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)

    x = Conv1D(cnn_filters[1], kernel_sizes[1], activation="relu",
               padding="same", name="conv2")(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)

    x = Conv1D(cnn_filters[2], kernel_sizes[2], activation="relu",
               padding="same", name="conv3")(x)

    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dropout(dropout, name="drop")(x)

    rul_output = Dense(1, activation="linear", name="rul")(x)
    health_output = Dense(n_classes, activation="softmax", name="health")(x)

    model = Model(inputs=inputs, outputs=[rul_output, health_output], name="CNNOnly")
    return model
