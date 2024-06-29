"""Represents a module containing the creation of the models that can be used in the training process."""
import torch

from action_recognition.src.nn.lstm import LSTM


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
