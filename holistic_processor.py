import cv2
import mediapipe as mp
from visualization import LandmarkDrawer

class HolisticProcessor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawer = LandmarkDrawer()

    def process_frame(self, frame):
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Pose detection
        pose_detection_results = self.holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        self.drawer.draw_landmarks(image, pose_detection_results)
        
        return image
