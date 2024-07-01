"""Represents a module containing the creation of the models that can be used in the training process."""
import torch

from action_recognition.src.nn.lstm import LSTM
from action_recognition.src.nn.transformer import ActionRecognitionTransformer


def create_lstm_model(
    input_size: int,
    hidden_sizes: list[int],
    fc_sizes: list[int],
    output_classes: int,
) -> torch.nn.Module:
    """Creates the LSTM model for the training process.

    Args:
        input_size (int): The number of input features.
        hidden_sizes (list[int]): A list containing the number of features in each LSTM layer.
        fc_sizes (list[int]): A list containing the number of features in each fully connected layer.
        output_classes (int): The number of classes between which the model has to differentiate.
    
    Returns:
        torch.nn.Module: The model for the training process.
    """
    model = LSTM(input_size=input_size, hidden_sizes=hidden_sizes, fc_sizes=fc_sizes, output_classes=output_classes)
    return model


def create_transformer_model(
    input_size: int,
    output_classes: int,
    model_dimension: int = 32, 
    number_heads: int = 2, 
    number_encoder_layers: int = 1, 
    number_decoder_layers: int = 1, 
    dimension_feedforward: int = 32
) -> torch.nn.Module:
    """Creates the Transformer model for the training process.

    Args:
        input_size (int): The number of input features.
        output_classes (int): The number of classes between which the model has to differentiate.
        model_dimension (int): The number of expected features in the encoder/decoder inputs.
        number_heads (int): The number of heads in the multiheadattention models.
        number_encoder_layers (int): The number of sub-encoder-layers in the encoder.
        number_decoder_layers (int): The number of sub-decoder-layers in the decoder.
        dimension_feedforward (int): The dimension of the feedforward network model.
    
    Returns:
        torch.nn.Module: The Transformer model for the training process.
    """
    model = ActionRecognitionTransformer(
        input_dim=input_size,
        num_classes=output_classes,
        model_dimension=model_dimension,
        number_heads=number_heads,
        number_encoder_layers=number_encoder_layers,
        number_decoder_layers=number_decoder_layers,
        dimension_feedforward=dimension_feedforward
    )
    return model
