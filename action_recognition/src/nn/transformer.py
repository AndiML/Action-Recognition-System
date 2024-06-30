"""Represents a module containing a Transformer network for action recognition on time series data."""
import torch
import torch.nn as nn

from action_recognition.src.nn.base_model import BaseModel


class Transformer(BaseModel):
    """Represents a Transformer model architecture for action recognition on time series data."""

    model_id = 'transformer'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int, 
        d_model: int = 32, 
        nhead: int = 2, 
        num_encoder_layers: int = 1, 
        num_decoder_layers: int = 1, 
        dim_feedforward: int = 32, 
        dropout: float = 0.1
    ) -> None:
        """Initializes a new Transformer instance.

        Args:
            input_dim (int): The dimension of the input vectors.
            num_classes (int): The number of classes between which the model has to differentiate.
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(Transformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout
        )

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            src (torch.Tensor): The input sequence of vectors.

        Returns:
            torch.Tensor: Returns the outputs of the model.
        """
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.positional_encoding(src)

        src_key_padding_mask = self.generate_padding_mask(src)

        transformer_output = self.transformer(
            src, src,
            src_key_padding_mask=src_key_padding_mask
        )

        output = self.fc_out(transformer_output[:, -1, :])
        return output

    def generate_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Generates padding mask for the input sequence.

        Args:
            src (torch.Tensor): The input sequence of vectors.

        Returns:
            torch.Tensor: Returns the padding mask.
        """
        # Assuming 0 is used for padding in the input sequences
        padding_mask = (src == 0).all(dim=-1).T
        return padding_mask  # Shape: [batch_size, seq_len]


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
