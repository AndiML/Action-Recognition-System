import mediapipe

class LandmarkDrawer:
    """
    A utility class for drawing landmarks on an image.
    Utilizes MediaPipe's drawing utilities and predefined drawing specifications.
    """

    def __init__(self):
        """
        Initializes the LandmarkDrawer with specific drawing specifications for different landmarks.
        """
        # Initialize drawing specifications with fixed colors for each component
        self.face_drawing_spec = mediapipe.solutions.drawing_utils.DrawingSpec(
            color=(255, 0, 0),
            thickness=1, 
            circle_radius=1
        ) 
        
        self.right_hand_drawing_spec = mediapipe.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        ) 
        
        self.left_hand_drawing_spec = mediapipe.solutions.drawing_utils.DrawingSpec(
            color=(0, 0, 255), 
            thickness=2,
            circle_radius=2
        ) 
       
        self.pose_drawing_spec = mediapipe.solutions.drawing_utils.DrawingSpec(
            color=(255, 255, 0), 
            thickness=2, 
            circle_radius=2,
        )  

        # Initialize MediaPipe drawing utils
        self.mp_drawing = mediapipe.solutions.drawing_utils

    def draw_landmarks(self, image, pose_detection_results):
        """
        Draws the landmarks on the given image based on the detection results.

        Args:
            image (numpy.ndarray): The image on which landmarks are to be drawn.
            pose_detection_results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The detection results containing landmarks.
        """
        # Draw face landmarks
        if pose_detection_results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                pose_detection_results.face_landmarks, 
                mediapipe.solutions.holistic.FACEMESH_CONTOURS,
                self.face_drawing_spec,
                self.face_drawing_spec
            )
        
        # Draw right hand landmarks
        if pose_detection_results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                pose_detection_results.right_hand_landmarks, 
                mediapipe.solutions.holistic.HAND_CONNECTIONS,
                self.right_hand_drawing_spec,
                self.right_hand_drawing_spec
            )

        # Draw left hand landmarks
        if pose_detection_results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                pose_detection_results.left_hand_landmarks, 
                mediapipe.solutions.holistic.HAND_CONNECTIONS,
                self.left_hand_drawing_spec,
                self.left_hand_drawing_spec
            )

        # Draw pose landmarks
        if pose_detection_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, pose_detection_results.pose_landmarks,
                mediapipe.solutions.holistic.POSE_CONNECTIONS,
                self.pose_drawing_spec,
                self.pose_drawing_spec
            )
    

