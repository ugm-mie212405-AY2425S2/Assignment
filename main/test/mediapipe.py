import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    self.drawing.draw_detection(frame, detection)

            cv2.imshow('Face Detection Only', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # def __del__(self):
    #     self.cap.release()
    #     cv2.destroyAllWindows()
    #     self.face_detection.close()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run
