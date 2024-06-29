"""A sub-package that contains models and algorithms for generating the model for training."""

from action_recognition.src.nn.lstm import LSTM

MODEL_IDS = [LSTM.model_id]
"""Contains the IDs of all available model architectures."""

DEFAULT_MODEL_ID = LSTM.model_id
"""Contains the ID of the default model architecture."""

__all__ = [
    'LSTM',
    'create_lstm_model',
    'MODEL_IDS',
    'DEFAULT_MODEL_ID'
]
