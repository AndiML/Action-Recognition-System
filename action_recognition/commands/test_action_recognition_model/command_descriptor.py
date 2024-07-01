"""Represents a module that contains the descriptor for the test action recognition model command."""

from argparse import ArgumentParser

from action_recognition.commands.base import BaseCommandDescriptor


class TestActionRecognitionModelCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of generate-action-data command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'test-action-recognition-model'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return '''Retrieves the user data based on the number and labels of the action to be detected'''

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        parser.add_argument(
            'path_to_training_directory',
            type=str,
            help='The path to the directory into which the results of the training were saved.'
        )

        parser.add_argument(
            '-g',
            '--use_gpu',
            action='store_true',
            help="If the switch is set, cuda is utilized for the federated learning process."
        )

