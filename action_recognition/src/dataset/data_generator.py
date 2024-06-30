"""Represents a module containing the creation of the data for the training process."""
import os
from typing import Tuple

import torch
import numpy


def create_dataset(dataset_path: str) -> Tuple[torch.tensor, torch.tensor]:
    """Creates the dataset for the training process.

    Args:
        dataset_path (str): The path to the dataset directory.

    Returns:
        Tuple[torch.tensor, torch.tensor]: The training data and labels.
    """
    actions = os.listdir(dataset_path)
    action_to_label = {action: idx for idx, action in enumerate(actions)}
    training_data = []
    labels = []

    for action in actions:
        action_dir = os.path.join(dataset_path, action)
        label = action_to_label[action]
        number_of_videos = len(os.listdir(action_dir))
        for video_number in range(1, number_of_videos + 1):
            video_dir = os.path.join(action_dir, f"video_{video_number}")
            if not os.path.exists(video_dir):
                continue
            time_series_for_video = []
            for frame_file in os.listdir(video_dir):
                if frame_file.endswith('.npy'):
                    frame_path = os.path.join(video_dir, frame_file)
                    keypoints = numpy.load(frame_path)
                    time_series_for_video.append(keypoints)
            training_data.append(time_series_for_video)
            labels.append(label)

    training_data = numpy.array(training_data)
    labels = numpy.array(labels)

    # Convert to tensor after conversion to numpy for performance enhancement
    training_data = torch.tensor(training_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return training_data, labels


