"""Represents a module containing a Transformer network for action recognition on time series data."""
import torch
import torch.nn as nn

from action_recognition.src.nn.base_model import BaseModel


class ActionRecognitionTransformer(BaseModel):
    """Represents a Transformer model architecture for action recognition on time series data."""

    model_id = 'transformer'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int, 
        model_dimension: int = 32, 
        number_heads: int = 2, 
        number_encoder_layers: int = 1, 
        number_decoder_layers: int = 1, 
        dimension_feedforward: int = 32, 
    ) -> None:
        """Initializes a new Transformer instance.

        Args:
            input_dim (int): The dimension of the input vectors.
            num_classes (int): The number of classes between which the model has to differentiate.
            model_dimension (int): The number of expected features in the encoder/decoder inputs.
            number_heads (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model.
        """
        super(ActionRecognitionTransformer, self).__init__()

        self.input_dim = input_dim
        self.model_dimension = model_dimension
        self.num_classes = num_classes

        self.embedding = nn.Linear(input_dim, model_dimension)
        self.positional_encoding = PositionalEncoding(model_dimension)

        self.transformer = nn.Transformer(
            model_dimension, number_heads, number_encoder_layers, number_decoder_layers, dimension_feedforward
        )

        self.fc_out = nn.Linear(model_dimension, num_classes)

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
        # Extracts information for the last time step
        output = self.fc_out(transformer_output[:, -1, :])
        return output

    def generate_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Generates padding mask for the input sequence.

        Args:
            src (torch.Tensor): The input sequence of vectors.

        Returns:
            torch.Tensor: Returns the padding mask.
        """
        padding_mask = (src == 0).all(dim=-1).T
        return padding_mask 


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    
    def __init__(self, model_dimension: int, maximum_length: int = 5000) -> None:
        """
        Initializes the PositionalEncoding instance.
        
        Args:
            model_dimension (int): The dimension of the model (number of expected features in the input).
            maximum_length (int): The maximum length of the sequences to be encoded. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(maximum_length, model_dimension)
        position = torch.arange(0, maximum_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor of shape (sequence_length, batch_size, model_dimension).
        
        Returns:
            torch.Tensor: The input tensor with positional encodings added, of the same shape as input.
        """
        x = x + self.pe[:x.size(0), :]
        return x
