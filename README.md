# Action Recognition Framework README

## Overview
The Action Recognition Framework is designed to facilitate the generation of action recognition data and the training of action recognition models. It comprises two main components: the `GenerateActionDataCommand` for generating training data and the `TrainActionRecognitionModelCommand` for training models. The framework is built using Python and leverages OpenCV for video capture, numpy for data manipulation, and PyTorch for model training.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Action Data](#generating-action-data)
  - [Training the Action Recognition Model](#training-the-action-recognition-model)
- [Components](#components)
  - [GenerateActionDataCommand](#generateactiondatacommand)
  - [TrainActionRecognitionModelCommand](#trainactionrecognitionmodelcommand)
- [Command Line Arguments](#command-line-arguments)
  - [GenerateActionDataCommand Arguments](#generateactiondatacommand-arguments)
  - [TrainActionRecognitionModelCommand Arguments](#trainactionrecognitionmodelcommand-arguments)
- [Logging](#logging)
- [License](#license)

## Installation
To install the Action Recognition Framework, clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-repo/action-recognition-framework.git
cd action-recognition-framework
pip install -r requirements.txt
```

## Usage

### Generating Action Data
To generate action data, use the `GenerateActionDataCommand`. This command captures video frames of specified actions and stores the extracted keypoints.

Example usage:
```bash
python -m action_recognition generate-action-data --output_path ./output --actions wave,clap --number_of_videos 10 --number_of_frames 24 --minimum_detection_confidence 0.5 --minimum_tracking_confidence 0.5
```

### Training the Action Recognition Model
To train an action recognition model, use the `TrainActionRecognitionModelCommand`. This command trains a model using the generated data and specified hyperparameters.

Example usage:
```bash
python -m action_recognition train-action-recognition-model --output_path ./output --dataset_path ./output/user_data --epochs 100 --batchsize 1 --learning_rate 0.001 --optimizer adam --use_gpu
```

## Components

### GenerateActionDataCommand
This component is responsible for capturing video frames and extracting keypoints for specified actions. It utilizes the `HolisticPoseProcessor` for keypoint extraction and stores the data in a structured directory.

#### Key Methods
- `__init__()`: Initializes the command.
- `run(command_line_arguments: Namespace)`: Executes the data generation process.

### TrainActionRecognitionModelCommand
This component handles the training of action recognition models. It supports both LSTM and Transformer models and can use either SGD or Adam optimizers.

#### Key Methods
- `__init__()`: Initializes the command.
- `run(command_line_arguments: Namespace)`: Executes the training process, including data preparation, model creation, training, evaluation, and plotting of training statistics.

## Command Line Arguments

### GenerateActionDataCommand Arguments
- `output_path` (str): The directory where the results are saved.
- `actions` (list of str): List of actions to capture.
- `number_of_videos` (int): Number of videos per action. Default is 10.
- `number_of_frames` (int): Number of frames per video. Default is 24.
- `minimum_detection_confidence` (float): Minimum detection confidence for the model. Default is 0.5.
- `minimum_tracking_confidence` (float): Minimum tracking confidence for the model. Default is 0.5.

### TrainActionRecognitionModelCommand Arguments
- `output_path` (str): The directory where the results are saved.
- `dataset_path` (str): Path to the dataset.
- `epochs` (int): Number of training epochs. Default is 100.
- `batchsize` (int): Batch size for training. Default is 1.
- `learning_rate` (float): Learning rate for the optimizer. Default is 0.001.
- `set_momentum` (float): Momentum for the optimizer. Default is 0.9.
- `weight_decay` (float): Weight decay for the optimizer. Default is 0.0005.
- `use_gpu` (bool): Flag to use GPU for training.
- `hidden_sizes` (list of int): Sizes of hidden layers. Default is [32, 64].
- `fc_sizes` (list of int): Sizes of fully connected layers. Default is [32].
- `optimizer` (str): Type of optimizer to use. Choices are 'sgd' and 'adam'. Default is 'sgd'.
- `model_type` (str): Type of model to use. Choices are 'lstm' and 'transformer'.
- `validation_split` (float): Fraction of data for validation. Default is 0.05.
- `test_split` (float): Fraction of data for testing. Default is 0.05.
- `patience` (int): Number of epochs to wait before early stopping. Default is 10.
- `model_dimension` (int): Number of expected features in encoder/decoder inputs. Default is 32.
- `number_heads` (int): Number of heads in multi-head attention models. Default is 2.
- `number_encoder_layers` (int): Number of sub-encoder-layers in the encoder. Default is 1.
- `dimension_feedforward` (int): Dimension of the feedforward network model. Default is 32.

## Logging
The framework uses Python's built-in logging module to provide detailed logs of the data generation and training processes. Logs include information about the creation of directories, frame capture statuses, model training progress, and more.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
