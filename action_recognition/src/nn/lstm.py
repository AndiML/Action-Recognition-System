"""Represents a module containing a Long Short-Term Memory (LSTM) network for the time series datasets."""
import torch
import torch.nn as nn

from action_recognition.src.nn.base_model import BaseModel


class LSTM(BaseModel):
    """Represents an LSTM model architecture for action recognition on time series data."""

    model_id = 'lstm'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(self, input_size: int, hidden_sizes: list[int], fc_sizes: list[int], output_classes: int) -> None:
        """Initializes a new LSTMCifar10 instance.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list[int]): A list containing the number of features in each LSTM layer.
            fc_sizes (list[int]): A list containing the number of features in each fully connected layer.
            output_classes (int): The number of classes between which the model has to differentiate.
        """
        super(LSTM, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_sizes[i], batch_first=True))

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_sizes)):
            input_dim = hidden_sizes[-1] if i == 0 else fc_sizes[i - 1]
            self.fc_layers.append(nn.Linear(input_dim, fc_sizes[i]))
        self.fc_layers.append(nn.Linear(fc_sizes[-1], output_classes))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.
        """
        batch_size = input_tensor.size(0)

        # Initialize hidden and cell states
        h0 = [torch.zeros(1, batch_size, size).to(input_tensor.device) for size in self.hidden_sizes]
        c0 = [torch.zeros(1, batch_size, size).to(input_tensor.device) for size in self.hidden_sizes]

        out = input_tensor
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out, (h0[i], c0[i]))

        # Decode the hidden state of the last time step
        out = out[:, -1, :,]

        # Pass through fully connected layers
        for fc in self.fc_layers:
            out = fc(out)

        return out
