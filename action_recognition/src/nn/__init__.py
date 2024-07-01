"""A sub-package that contains models and algorithms for generating the model for training."""

from action_recognition.src.nn.lstm import LSTM
from action_recognition.src.nn.transformer import Transformer

MODEL_IDS = [LSTM.model_id, Transformer.model_id]
"""Contains the IDs of all available model architectures."""

DEFAULT_MODEL_ID = LSTM.model_id
"""Contains the ID of the default model architecture."""

__all__ = [
    'LSTM',
    'create_lstm_model',
    'create_transformer_model'
    'MODEL_IDS',
    'DEFAULT_MODEL_ID'
]
