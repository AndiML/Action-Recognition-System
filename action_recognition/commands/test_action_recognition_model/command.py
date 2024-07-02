"""Represents a module that contains the test action recognition model command."""

import logging
import os
import yaml
from argparse import Namespace

import numpy
import cv2
import torch

from action_recognition.commands.base import BaseCommand
from action_recognition.src.training import ModelTrainer
from action_recognition.src.holistic_processor import HolisticPoseProcessor
from action_recognition.src.nn.model_generator import create_lstm_model, create_transformer_model


class TestActionRecognitionModelCommand(BaseCommand):
    """Represents a command that represents the data generation process."""

    def __init__(self) -> None:
        """Initializes a new TestActionRecognitionModel instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        # Selects device for inference
        device = 'cuda' if command_line_arguments.use_gpu else 'cpu'
        self.logger.info("Selected %s for Inference", device.upper(), extra={'start_section': True})

        # Loads the hyperparamters
        hyperparameters_filepath = os.path.join(command_line_arguments.path_to_training_directory, 'hyperparameters.yaml')
        with open(hyperparameters_filepath, 'r', encoding='utf-8') as file:
            hyperparameters = yaml.load(file, Loader=yaml.FullLoader)
        
        model_parameters = hyperparameters['command_line_arguments']
        actions_to_index = hyperparameters['action']
        index_to_actions = {index: action for action, index in actions_to_index.items()}
        
        # Loads the models
        self.logger.info('Creating model of type %s', model_parameters['model_type'], extra={'start_section': True})
        if model_parameters['model_type'] == 'lstm':
            model = create_lstm_model(
                input_size=model_parameters['input_dimension'],
                hidden_sizes=model_parameters['hidden_sizes'], 
                fc_sizes=model_parameters['fc_sizes'], 
                output_classes=len(actions_to_index.keys())
            )
        elif model_parameters['model_type'] == 'transformer':
            model = create_transformer_model(
                input_size=model_parameters['input_dimension'],
                model_dimension=model_parameters['model_dimension'],
                number_heads=model_parameters['number_heads'],
                number_encoder_layers=model_parameters['number_encoder_layers'],
                dimension_feedforward=model_parameters['dimension_feedforward'],
                output_classes=len(actions_to_index.keys())
            )
        else:
            exit('Architecture not supported.')
        self.logger.info(f"Using {model_parameters['model_type'].capitalize()} for Training", extra={'start_section': True})
        
        # Loads the model from the training
        self.logger.info('Loaded trained model state', extra={'start_section': True})
        model.load_state_dict(torch.load(os.path.join(command_line_arguments.path_to_training_directory, 'best_model.pth')))
        
        self.logger.info('Load holistic model for Landmark Extraction', extra={'start_section': True})
        # Initializes the holistic model processor
        holistic_model = HolisticPoseProcessor(
            min_detection_confidence=model_parameters['minimum_detection_confidence'], 
            min_tracking_confidence=model_parameters['minimum_tracking_confidence']
        )

        # Uses the model trainer class  for inference
        model_inference = ModelTrainer(
            output_path='./',
            model=model,
            loss_function=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            device=device,
            logger=self.logger,
            training_data=torch.ones(1, 1, 1),
            labels=torch.ones(1, 1, 1)
        )
        
        # Helper variables for inference
        # The sequence_length determines the number of frames to consider for making a single prediction
        sequence_length = model_parameters['number_of_frames']

        # The consistency_check_length helps to ensure that an action is consistently predicted over a number of frames, 
        # reducing the false positive detection rate
        consistency_check_length = 3

        # Holds the keypoints for each frame that will be used for prediction
        processed_frames = []

        # Contains the predictions made after processing a specific number of frames, as determined by the sequence_length
        predictions = []

        last_action = None
        last_probability = 0.0
        action_probability = 0.0

        # OpenS the video capture to start reading from the webcam
        capture = cv2.VideoCapture(0)
        self.logger.info('Starting video capture', extra={'start_section': True})

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                self.logger.warning("Frame capture failed", extra={'start_section': True})
                continue

            # Extract keypoints from the frame using the holistic model
            _, results = holistic_model.process_frame(frame)
            keypoints = holistic_model.extract_all_keypoints(results)
            processed_frames.append(keypoints)
         
            if len(processed_frames) == sequence_length:
                predicted_action_id, predicted_probability_for_action_id = model_inference.predict_action(
                    numpy.array(processed_frames)
                )
                predictions.append(predicted_action_id)
                processed_frames = []
                
                recent_predictions = predictions[-consistency_check_length:]
                if numpy.unique(recent_predictions)[0] == predicted_action_id:
                    action_to_string = index_to_actions[predicted_action_id]
                    action_probability = predicted_probability_for_action_id
                    last_action = action_to_string
                    last_probability = action_probability
                    self.logger.info(f"Detected action: {last_action} with probability {last_probability:.2f}", extra={'start_section': True})
             
            # Display the action name and probability on the frame
            if last_action is not None:
                cv2.putText(
                    frame, 
                    f'Action: {last_action} ({last_probability:.2f})', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0, 255, 0), 
                    2,
                    cv2.LINE_AA
                )
            cv2.imshow('Action Recognition', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break

        capture.release()
        cv2.destroyAllWindows()