from typing import Tuple
from matplotlib import pyplot as plt
import torch
import torch.utils.data


class ModelTrainer:
    def __init__(
        self,
        output_path: str,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        logger,
        training_data: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1,
        validation_split: float = 0.05
    ) -> None:
        """
        Initializes a new ModelTrainer instance.

        Args:
            output_path (str): Path to save the best model and training statistics.
            model (torch.nn.Module): The neural network model to train.
            criterion (torch.nn.Module): Loss function to calculate loss.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            device (str): The device where training and inference are carried out.
            logger: Logger for logging training information.
            training_data (torch.Tensor): Tensor containing the training data.
            labels (torch.Tensor): Tensor containing the labels corresponding to the training data.
            batch_size (int): Batch size for DataLoader (default: 1).
            validation_split (float): Fraction of the data to be used as validation set (default: 0.05).
        """
        self.output_path = output_path
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.model.to(self.device)
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []
        self.train_loader, self.val_loader = self.create_dataloaders(training_data, labels, batch_size, validation_split)

    def create_dataloaders(
        self,
        training_data: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1,
        validation_split: float = 0.05
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create PyTorch DataLoader instances for training and validation.

        Args:
            training_data (torch.Tensor): Tensor containing the training data.
            labels (torch.Tensor): Tensor containing the labels corresponding to the training data.
            batch_size (int): Batch size for DataLoader (default: 1).
            validation_split (float): Fraction of the data to be used as validation set (default: 0.05).

        Returns:
            train_loader (torch.utils.data.DataLoader): DataLoader for training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation set.
        """
        dataset = torch.utils.data.TensorDataset(training_data, labels)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate the model on the validation set.

        Returns:
            val_loss (float): Average loss on the validation set.
            accuracy (float): Accuracy on the validation set.
        """
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        val_loss /= total_predictions
        accuracy = correct_predictions / total_predictions
        return val_loss, accuracy

    def train_model(
        self,
        num_epochs: int = 100,
        patience: int = 10
    ) -> torch.nn.Module:
        """
        Train the neural network model.

        Args:
            num_epochs (int): Number of epochs to train (default: 100).
            patience (int): Number of epochs to wait before early stopping (default: 10).

        Returns:
            model (torch.nn.Module): Trained model with best validation performance.
        """
        best_loss = float('inf')
        patience_counter = 0
        number_of_datapoints = 0
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                number_of_datapoints += labels.size(0)

            epoch_loss = running_loss / number_of_datapoints
            val_loss, val_accuracy = self.evaluate_model()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            self.logger.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info("Early stopping triggered")
                break

        self.model.load_state_dict(torch.load(self.output_path))
        return self.model

    def plot_metrics(self) -> None:
        """Plot the validation loss and validation accuracy."""
        epochs = range(1, len(self.val_losses) + 1)

        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.val_losses, 'b', label='Validation loss')
        plt.title('Validation loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_accuracies, 'b', label='Validation accuracy')
        plt.title('Validation accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        