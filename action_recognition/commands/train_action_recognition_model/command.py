"""Represents a module that contains the training of the action recognition model command."""

import logging
from argparse import Namespace

import numpy
import torch

from action_recognition.commands.base import BaseCommand
from action_recognition.src.training import ModelTrainer
from action_recognition.src.dataset.data_generator import create_dataset
from action_recognition.src.nn.model_generator import create_lstm_model


class TrainActionRecognitionModelCommand(BaseCommand):
    """Represents a command that represents the train action recognition model process."""

    def __init__(self) -> None:
        """Initializes a new TrainActionRecognitionModel instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        # Selects device for training
        device = 'cuda' if command_line_arguments.use_gpu else 'cpu'
        self.logger.info("Selected %s for  Training Process", device.upper())
        
        # Creates training data from actions captured as video frames
        training_data, labels = create_dataset(command_line_arguments.dataset_path)
        self.logger.info('Created Training Data from Action in Video Frames', extra={'start_section': True})
    
        # Creates the model for training 
        if command_line_arguments.model_type == 'lstm':
            model = create_lstm_model(
                input_size=training_data.shape[2],
                hidden_sizes=command_line_arguments.hidden_sizes, 
                fc_sizes=command_line_arguments.fc_sizes, 
                output_classes=len(numpy.unique(labels))
            )
        else:
            exit('Not Supported')
        self.logger.info(f'Using {command_line_arguments.model_type.capitalize()} for Training', extra={'start_section': True})
        
        # Creates the optimizer for training
        optimizer_kind = command_line_arguments.optimizer
        optimizer: torch.optim.SGD | torch.optim.Adam
        if optimizer_kind == 'sgd':
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=command_line_arguments.learning_rate, 
                momentum=command_line_arguments.set_momentum, 
                weight_decay=command_line_arguments.weight_decay
            )
        elif optimizer_kind == 'adam':
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=command_line_arguments.learning_rate, 
                weight_decay=command_line_arguments.weight_decay
            )
        else:
            raise ValueError(f'The optimizer "{optimizer_kind}" is not supported.')

        # Setup loss and optimizer
        loss_function = torch.nn.CrossEntropyLoss()

        # Create ModelTrainer instance and train the model
        trainer = ModelTrainer(
            output_path=command_line_arguments.output_path,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            logger=self.logger,
            training_data=training_data,
            labels=labels,
            batch_size=command_line_arguments.batchsize,
            validation_split=command_line_arguments.validation_split,
            test_split=command_line_arguments.test_split
        )

        trainer.train_model(num_epochs=command_line_arguments.epochs)
        self.logger.info('Finished Training', extra={'start_section': True})
        trainer.evaluate_on_test_set()
        
        # Plots the statistics of the training process
        trainer.plot_metrics()





    
       




