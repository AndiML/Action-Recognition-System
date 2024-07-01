"""Represents a module that contains the descriptor for the training of the action recognition command."""

from argparse import ArgumentParser

from action_recognition.commands.base import BaseCommandDescriptor
from action_recognition.src.nn import DEFAULT_MODEL_ID, MODEL_IDS


class TrainActionRecognitionModelCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of the training of the action recognition command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'train-action-recognition-model'

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
            'dataset_path',
            type=str,
            help='The path of the dataset.'
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

        parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=100,
            help="The number of epochs utilized during training."
        )
        parser.add_argument(
            '-b',
            '--batchsize',
            type=int,
            default=1,
            help="Batch size during training."
        )
        parser.add_argument(
            '-l',
            '--learning_rate',
            type=float,
            default=0.001,
            help='Learning rate utilized during training.'
        )
        parser.add_argument(
            '-m',
            '--set_momentum',
            type=float,
            default=0.9,
            help='Sets the level of momentum for specified optimizer'
        )
        parser.add_argument(
            '-W',
            '--weight_decay',
            type=float,
            default=0.0005,
            help='The rate at which the weights are decayed during optimization. Defaults to 0.0005.'
        )

        parser.add_argument(
            '-g',
            '--use_gpu',
            action='store_true',
            help="If the switch is set, cuda is utilized for the federated learning process."
        )

        parser.add_argument(
            '-H',
            '--hidden_sizes',
            type=int,
            nargs='+',
            default=[32, 64],
            help='List of hidden layer sizes. Defaults to [32, 64].'
        ) 

        parser.add_argument(
            '-c',
            '--fc_sizes',
            type=int,
            nargs='+',
            default=[32],
            help='List of fully connected layer sizes. Defaults to [32].'
        )

        parser.add_argument(
            '-o',
            '--optimizer',
            type=str,
            default='sgd',
            choices=['sgd', 'adam'],
            help="Type of optimizer"
        )

        parser.add_argument(
            '-M',
            '--model_type',
            type=str,
            default=DEFAULT_MODEL_ID,
            choices=MODEL_IDS,
            help='Type of neural network architecture used for training. '
        )

        parser.add_argument(
            '-V',
            '--validation_split',
            type=float,
            default=0.05,
            help='Fraction of the data to be used as validation set. Default to 0.05.'
        )

        parser.add_argument(
            '-T',
            '--test_split', 
            type=float, 
            default=0.05, 
            help='Fraction of the data to be used as test set. Default to 0.05.'
        )

        parser.add_argument(
            '-p',
            '--patience', 
            type=int, 
            default=10, 
            help='Number of epochs to wait before early stopping if validation loss does not improve. Default to 10.'
        )

        parser.add_argument(
            'O',
            '--model_dimension',
            type=int,
            default=32,
            help='The number of expected features in the encoder/decoder inputs. Default to 32.'
        )
        # For Transformer model
        parser.add_argument(
            '-n',
            '--number_heads',
            type=int,
            default=2,
            help='The number of heads in the multiheadattention models. Default to 2.'
        )

        parser.add_argument(
            '-E',
            '--number_encoder_layers',
            type=int,
            default=1,
            help='The number of sub-encoder-layers in the encoder. Default to 1.'
        )

        parser.add_argument(
            '-D',
            '--number_decoder_layers',
            type=int,
            default=1,
            help='The number of sub-decoder-layers in the decoder. Default to 1.'
        )

        parser.add_argument(
            '-d',
            '--dimension_feedforward',
            type=int,
            default=32,
            help='The dimension of the feedforward network model. Default to 32.'
        )
