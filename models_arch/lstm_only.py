"""
models_arch/lstm_only.py
------------------------
LSTM-only model with dual output heads (for model comparison).
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units: tuple[int, int, int] = (64, 64, 32),
    dropout: float = 0.2,
    n_classes: int = 3,
) -> Model:
    """
    LSTM-only model with three stacked LSTM layers and dual output heads.

    Parameters
    ----------
    input_shape : (window_size, n_features)
    lstm_units  : hidden units for each LSTM layer (3 layers)
    dropout     : dropout rate
    n_classes   : number of health classes

    Returns
    -------
    model : compiled Keras Model with two outputs ['rul', 'health']
    """
    inputs = Input(shape=input_shape, name="sensor_sequence")

    x = LSTM(lstm_units[0], activation="tanh", return_sequences=True, name="lstm1")(inputs)
    x = Dropout(dropout, name="drop1")(x)
    x = LSTM(lstm_units[1], activation="tanh", return_sequences=True, name="lstm2")(x)
    x = Dropout(dropout, name="drop2")(x)
    x = LSTM(lstm_units[2], activation="tanh", return_sequences=False, name="lstm3")(x)
    x = Dropout(dropout, name="drop3")(x)

    rul_output = Dense(1, activation="linear", name="rul")(x)
    health_output = Dense(n_classes, activation="softmax", name="health")(x)

    model = Model(inputs=inputs, outputs=[rul_output, health_output], name="LSTMOnly")
    return model
