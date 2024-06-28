"""Represents a module that contains the descriptor for the federated-averaging command."""

from argparse import ArgumentParser

from action_recognition.commands.base import BaseCommandDescriptor


class DataGeneratorCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of federated averaging algorithm command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'data-generator'

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
            'output_path',
            type=str,
            help='The path to the directory into which the results of the experiments are saved.'
        )

        parser.add_argument(
            'actions',
            type=str,
            nargs='+',
            help='The the actions as strings the user want to create data for.'
        )
