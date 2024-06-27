import mediapipe as mp

class LandmarkDrawer:
    def __init__(self):
        # Initialize drawing specifications with fixed colors for each component
        self.face_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 0, 0), thickness=1, circle_radius=1)  # Red for face
        self.right_hand_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)  # Green for right hand
        self.left_hand_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2)  # Blue for left hand
        self.pose_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 255, 0), thickness=3, circle_radius=3)  # Yellow for pose

        # Initialize MediaPipe drawing utils and holistic model
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

    def draw_landmarks(self, image, pose_detection_results):
        # Draw face landmarks
        if pose_detection_results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, pose_detection_results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                self.face_drawing_spec, self.face_drawing_spec)
        
        # Draw right hand landmarks
        if pose_detection_results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, pose_detection_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.right_hand_drawing_spec, self.right_hand_drawing_spec)

        # Draw left hand landmarks
        if pose_detection_results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, pose_detection_results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.left_hand_drawing_spec, self.left_hand_drawing_spec)

        # Draw pose landmarks
        if pose_detection_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, pose_detection_results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.pose_drawing_spec, self.pose_drawing_spec)
