"""Represents a module that contains the descriptor for the generate-action-data command."""

from argparse import ArgumentParser

from action_recognition.commands.base import BaseCommandDescriptor


class GenerateActionDataCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of generate-action-data command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'generate-action-data'

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
        parser.add_argument(
            '-v',
            '--number_of_videos',
            type=int,
            default=10,
            help="The number of video that will be created for each action. Defaults to 10."
        )

        parser.add_argument(
            '-f',
            '--number_of_frames',
            type=int,
            default=24,
            help="The number of frames each utilizes for capturing a specific action. Default to 24."
        )

        parser.add_argument(
            '-d',
            '--minimum_detection_confidence',
            type=float,
            default=0.5,
            help="Minimum confidence value ([0.0, 1.0]) from the person-detection model for the detection to be considered successful. Default to 0.5."
        )

        parser.add_argument(
            '-t',
            '--minimum_tracking_confidence',
            type=float,
            default=0.5,
            help="""Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the pose landmarks to be considered tracked successfully.
                Default to 0.5."""
        )

