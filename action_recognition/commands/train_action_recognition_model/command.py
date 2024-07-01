"""Represents a module that contains the training of the action recognition model command."""

import logging
from argparse import Namespace
import os
import yaml

import numpy
import torch

from action_recognition.commands.base import BaseCommand
from action_recognition.src.training import ModelTrainer
from action_recognition.src.dataset.data_generator import create_dataset
from action_recognition.src.nn.model_generator import create_lstm_model, create_transformer_model


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
        self.logger.info('Creating training data from dataset at %s', command_line_arguments.dataset_path)
        training_data, labels, action_to_label = create_dataset(command_line_arguments.dataset_path)
        self.logger.info('Created Training Data from Action in Video Frames')

        self.logger.info('Saves Hyperparameters into a YAML File')
        # Add input dimension directly to command line arguments
        command_line_arguments.input_dimension = training_data.shape[2]
        # Create a combined dictionary to store both command line arguments and my_dictionary under a specific key
        combined_data = {
            'command_line_arguments': vars(command_line_arguments),
            'action': action_to_label,
        }
        with open(os.path.join(command_line_arguments.output_path, 'hyperparameters.yaml'), 'w', encoding='utf-8') as hyperparameters_file:
            yaml.dump(combined_data, hyperparameters_file, default_flow_style=False)
        self.logger.info('Finished Saving Hyperparameters into a YAML File')
      
        # Creates the model for training 
        self.logger.info('Creating model of type %s', command_line_arguments.model_type)
        if command_line_arguments.model_type == 'lstm':
            model = create_lstm_model(
                input_size=training_data.shape[2],
                hidden_sizes=command_line_arguments.hidden_sizes, 
                fc_sizes=command_line_arguments.fc_sizes, 
                output_classes=len(numpy.unique(labels))
            )
        elif command_line_arguments.model_type == 'transformer':
            model = create_transformer_model(
                input_size=training_data.shape[2],
                model_dimension=command_line_arguments.model_dimension,
                number_heads=command_line_arguments.number_heads,
                number_encoder_layers=command_line_arguments.number_encoder_layers,
                dimension_feedforward=command_line_arguments.dimension_feedforward,
                output_classes=len(numpy.unique(labels))
            )
        else:
            exit('Architecture not supported.')
        self.logger.info(f'Using {command_line_arguments.model_type.capitalize()} for Training', extra={'start_section': True})
      
        # Creates the optimizer for training
        optimizer_kind = command_line_arguments.optimizer
        optimizer: torch.optim.SGD | torch.optim.Adam
        self.logger.info('Creating optimizer of type %s', optimizer_kind)
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
            self.logger.error('The optimizer %s is not supported', optimizer_kind)
            raise ValueError(f'The optimizer "{optimizer_kind}" is not supported.')

        # Setup loss and optimizer
        self.logger.info('Setting up loss function')
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
        self.logger.info('Starting training for %d epochs with patience of %d epochs', command_line_arguments.epochs, command_line_arguments.patience)
        trainer.train_model(num_epochs=command_line_arguments.epochs)
        self.logger.info('Finished Training', extra={'start_section': True})
        
        # Evaluates the trained model on the test set
        self.logger.info('Evaluating on test set')
        trainer.evaluate_on_test_set()
        
        # Plots the statistics of the training process
        self.logger.info('Plotting training statistics')
        trainer.plot_metrics()
        self.logger.info('Created statistics of the training process')





    
       




