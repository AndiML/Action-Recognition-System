import cv2
from holistic_processor import HolisticProcessor



class Application:
    """Represents the application of the post estimation pipeline."""

    def run(self) -> None:
        holistic_processor = HolisticProcessor()
        
        # Retrieve webcam feed
        capture = cv2.VideoCapture(0)

        while capture.isOpened():
            _, frame = capture.read()
            
            # Process frame
            processed_frame = holistic_processor.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow("Processed Webcam Feed", processed_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
