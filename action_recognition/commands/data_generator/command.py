"""Represents a module that contains the vanilla federated averaging command."""

import logging
import os
from argparse import Namespace

import numpy
import cv2


from action_recognition.commands.base import BaseCommand
from action_recognition.src.holistic_processor import HolisticPoseProcessor


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
        cap = cv2.VideoCapture(0)
        data_directory = os.path.join(command_line_arguments.output_path, 'user_data')
        os.makedirs(data_directory, exist_ok=True)
        self.logger.info('Created Directory for user data', extra={'start_section': True})
        holistic_model = HolisticPoseProcessor(
            min_detection_confidence=command_line_arguments.minimum_detection_confidence, 
            min_tracking_confidence=command_line_arguments.minimum_tracking_confidence
        )
        
        for action in command_line_arguments.actions:
            action_dir = os.path.join(data_directory, action)
            os.makedirs(action_dir, exist_ok=True)

            for video_number in range(1, command_line_arguments.number_of_videos + 1):
                video_dir = os.path.join(action_dir, f"video_{video_number}")
                os.makedirs(video_dir, exist_ok=True)
                
                # Display countdown before the first video for each action
                if video_number == 1:
                    for count in range(3, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            self.logger.warning(f"Frame capture failed for countdown")
                            continue
                        cv2.putText(
                            frame,
                            f'Starting Data Acquisition for Action{action.capitalize()} in {count}...',
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, 
                            (0, 0, 0), 
                            2, 
                            cv2.LINE_AA
                        )
                        cv2.imshow('Action Frame', frame)
                        if count !=0:
                            cv2.waitKey(1000)
        
                for frame_number in range(1, command_line_arguments.number_of_frames + 1):
    
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning(f"Frame capture failed for sequence {video_number}, frame {frame_number}")
                        continue

                    image, results = holistic_model.process_frame(frame)
                   
                    cv2.putText(
                        image,
                        f'Collecting Frames for {action.capitalize()} Video Number {video_number}',
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1, 
                        cv2.LINE_AA
                    )
                    cv2.imshow('Action Frame', image)

                    keypoints = holistic_model.extract_all_keypoints(results)
                    path_for_keypoints = os.path.join(video_dir, f"{frame_number}.npy")
                    numpy.save(path_for_keypoints, keypoints)
                    
                    # Quits process by user with Esc
                    if cv2.waitKey(10) & 0xFF == 27:
                        break
                self.logger.info(f'Finished Data Generation for Action {action.capitalize()} and Video {video_number}', extra={'start_section': True})
                
                # Display message before the next video
                if video_number < command_line_arguments.number_of_videos:
                    for count in range(3, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            self.logger.warning(f"Frame capture failed for next video countdown")
                            continue
                        cv2.putText(
                            frame,
                            f'Next Video for {action.capitalize()} Starting in {count}...',
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, 
                            (0, 0, 0), 
                            2, 
                            cv2.LINE_AA
                        )
                        cv2.imshow('Action Frame', frame)
                        if count != 0:
                            cv2.waitKey(1000)
        cap.release()
        cv2.destroyAllWindows()