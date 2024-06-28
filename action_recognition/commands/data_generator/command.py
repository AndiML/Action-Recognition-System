"""Represents a module that contains the vanilla federated averaging command."""

import logging
import os
import copy
from datetime import datetime
from argparse import Namespace
import random

import numpy


from action_recognition.commands.base import BaseCommand


class DataGeneratorCommand(BaseCommand):
    """Represents a command that represents the data generation process."""

    def __init__(self) -> None:
        """Initializes a new DataGenerator instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """

        for action in command_line_arguments.actions:
            action_dir = os.path.join(command_line_arguments.output_path, action)
            os.makedirs(action_dir, exist_ok=True)
            self.logger.info(f'Created Director for action {action}', extra={'start_section': True})

            for video_idx in range(1, 4):  # Example: create 3 video subdirectories
                video_dir = os.path.join(action_dir, f"video_{video_idx}")
                os.makedirs(video_dir, exist_ok=True)
                print(f"Created video subdirectory: {video_dir}")

                for frame_idx in range(1, 6):  
                    frame = numpy.random.rand(480, 640, 3) 
                    frame_path = os.path.join(video_dir, f"frame_{frame_idx}.npy")
                    numpy.save(frame_path, frame)
                    print(f"Saved frame {frame_idx} to {frame_path}")













        