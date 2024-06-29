"""A module that contains the abstract base class for models."""

from abc import ABC, abstractmethod
import torch


class BaseModel(ABC, torch.nn.Module):
    """Represents the abstract base class for all models."""
    def __init__(self) -> None:
        """_Initializes a BaseModel instance."""

        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.
        Returns:
            torch.Tensor: Returns the outputs of the model.
        """
        raise NotImplementedError
