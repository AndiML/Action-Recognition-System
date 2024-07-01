import cv2
import mediapipe
import numpy

from action_recognition.src.visualization import LandmarkDrawer


class HolisticPoseProcessor:
    """
    Represents a processor for holistic pose detection and landmark drawing.
    Utilizes MediaPipe's holistic solution for pose detection and a custom drawer for landmarks.
    """
    
    # Class variables for the dimensions of the individual landmarks
    FACE_LANDMARKS = 468 * 3
    HAND_LANDMARKS = 21 * 3
    POSE_LANDMARKS = 33 * 4

    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float):
        """
        Initializes the HolisticPoseProcessor with the specified detection and tracking confidence levels.

        Args:
            min_detection_confidence (float): Minimum confidence value for detection.
            min_tracking_confidence (float): Minimum confidence value for tracking.
        """
        self.holistic_model = mediapipe.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmark_drawer = LandmarkDrawer()

    def process_frame(self, frame: numpy.ndarray):
        """
        Processes a single frame to detect poses and draw landmarks.

        Args:
            frame (numpy.ndarray): The input frame in BGR format.

        Returns:
            numpy.ndarray: The processed frame with landmarks drawn.
        """

        # Convert the frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Optimization: set image as non-writeable

        # Perform pose detection
        pose_detection_results = self.holistic_model.process(image)
        image.flags.writeable = True  # Set image back to writeable

        # Convert the image back to BGR format for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks on the image
        self.landmark_drawer.draw_landmarks(image, pose_detection_results)
        
        return image, pose_detection_results
    
    def extract_face_keypoints(self, results) -> numpy.ndarray:
        """
        Extracts the face keypoints from the detection results.

        Args:
            results (HolisticResults): The detection results from MediaPipe's holistic model.

        Returns:
            numpy.ndarray: A flattened array of the face keypoints.
        """
        return numpy.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else numpy.zeros(self.FACE_LANDMARKS)

    def extract_left_hand_keypoints(self, results) -> numpy.ndarray:
        """
        Extracts the left hand keypoints from the detection results.

        Args:
            results (HolisticResults): The detection results from MediaPipe's holistic model.

        Returns:
            np.ndarray: A flattened array of the left hand keypoints.
        """
        return numpy.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else numpy.zeros(self.HAND_LANDMARKS)

    def extract_right_hand_keypoints(self, results) -> numpy.ndarray:
        """
        Extracts the right hand keypoints from the detection results.

        Args:
            results (HolisticResults): The detection results from MediaPipe's holistic model.

        Returns:
            np.ndarray: A flattened array of the right hand keypoints.
        """
        return numpy.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else numpy.zeros(self.HAND_LANDMARKS)

    def extract_pose_keypoints(self, results) -> numpy.ndarray:
        """
        Extracts the pose keypoints from the detection results.

        Args:
            results (HolisticResults): The detection results from MediaPipe's holistic model.

        Returns:
            np.ndarray: A flattened array of the pose keypoints.
        """
        return numpy.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else numpy.zeros(self.POSE_LANDMARKS)

    def extract_all_keypoints(self, results) -> numpy.ndarray:
        """
        Extracts all keypoints from the detection results and concatenates them into a single array.

        Args:
            results (HolisticResults): The detection results from MediaPipe's holistic model.

        Returns:
            np.ndarray: A flattened array of all keypoints from the face, pose, left hand, and right hand.
        """
        face_landmarks = self.extract_face_keypoints(results)
        left_hand_landmarks = self.extract_left_hand_keypoints(results)
        right_hand_landmarks = self.extract_right_hand_keypoints(results)
        pose_landmarks = self.extract_pose_keypoints(results)
        return numpy.concatenate([face_landmarks, left_hand_landmarks, right_hand_landmarks, pose_landmarks])
