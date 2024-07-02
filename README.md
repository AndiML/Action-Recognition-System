# Action Recognition Framework

## Overview
The Action Recognition Framework is designed to facilitate the generation of action recognition data and the training of action recognition models. It comprises three main components: the `GenerateActionDataCommand` for generating training data, the `TrainActionRecognitionModelCommand` for training models, and the `TestActionRecognitionModelCommand` for inference. The framework is built using Python and leverages OpenCV for video capture, numpy for data manipulation, and PyTorch for model training and inference.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Action Data](#generating-action-data)
  - [Training the Action Recognition Model](#training-the-action-recognition-model)
  - [Inference Using the Action Recognition Model](#inference-using-the-action-recognition-model)
- [Command Line Arguments](#command-line-arguments)
  - [GenerateActionDataCommand Arguments](#generateactiondatacommand-arguments)
  - [TrainActionRecognitionModelCommand Arguments](#trainactionrecognitionmodelcommand-arguments)
  - [TestActionRecognitionModelCommand Arguments](#testactionrecognitionmodelcommand-arguments)
- [Logging](#logging)
- [License](#license)

## Installation
To install the Action Recognition Framework, clone the repository and install the required dependencies:
```bash
git clone git@github.com:AndiML/Generated-Action-Recognition.git
cd Generated-Action-Recognition
pip install -r requirements.txt
```

## Usage

### Generating Action Data
To generate action data, use the `GenerateActionDataCommand`. This command captures video frames of specified actions and stores the extracted keypoints.

Example usage:
```bash
python -m action_recognition generate-action-data --output_path ./output --actions wave clap --number_of_videos 10 --number_of_frames 24 --minimum_detection_confidence 0.5 --minimum_tracking_confidence 0.5
```

### Training the Action Recognition Model
To train an action recognition model, use the `TrainActionRecognitionModelCommand`. This command trains a model using the generated data and specified hyperparameters. You can choose between LSTM and Transformer models, and specify various training parameters.

#### LSTM Model
Example usage:
```bash
python -m action_recognition train-action-recognition-model --output_path ./output --dataset_path ./output/user_data --epochs 100 --batchsize 1 --learning_rate 0.001 --optimizer adam --use_gpu --model_type lstm --hidden_sizes 32 64 --fc_sizes 32 --validation_split 0.05 --test_split 0.05 --patience 10 --minimum_detection_confidence 0.5 --minimum_tracking_confidence 0.5
```

Parameters:
- `hidden_sizes` (list of int): Sizes of hidden layers. Default is [32, 64].
- `fc_sizes` (list of int): Sizes of fully connected layers. Default is [32].

#### Transformer Model
Example usage:
```bash
python -m action_recognition train-action-recognition-model --output_path ./output --dataset_path ./output/user_data --epochs 100 --batchsize 1 --learning_rate 0.001 --optimizer adam --use_gpu --model_type transformer --model_dimension 32 --number_heads 2 --number_encoder_layers 1 --dimension_feedforward 32 --validation_split 0.05 --test_split 0.05 --patience 10 --minimum_detection_confidence 0.5 --minimum_tracking_confidence 0.5
```

Parameters:
- `model_dimension` (int): Number of expected features in encoder/decoder inputs. Default is 32.
- `number_heads` (int): Number of heads in multi-head attention models. Default is 2.
- `number_encoder_layers` (int): Number of sub-encoder-layers in the encoder. Default is 1.
- `dimension_feedforward` (int): Dimension of the feedforward network model. Default is 32.

### Inference Using the Action Recognition Model
To perform inference using a trained action recognition model, use the `TestActionRecognitionModelCommand`. This command uses the trained model to recognize actions in real-time from a video feed.

Example usage:
```bash
python -m action_recognition test-action-recognition-model --path_to_training_directory ./output --use_gpu
```

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
- `minimum_detection_confidence` (float): Minimum detection confidence for the holistic model. Default is 0.5.
- `minimum_tracking_confidence` (float): Minimum tracking confidence for the holistic model. Default is 0.5.

### TestActionRecognitionModelCommand Arguments
- `path_to_training_directory` (str): The path to the directory into which the results of the training were saved.
- `use_gpu` (bool): If the switch is set, GPU is utilized for inference.

## Logging
The framework uses Python's built-in logging module to provide detailed logs of the data generation, training, and inference processes. Logs include information about the creation of directories, frame capture statuses, model training progress, and more.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
