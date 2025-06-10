import cv2
import mediapipe as mp
from image_classifier import FaceNetClassifier  

def main():
    classifier = FaceNetClassifier(model_path='model/test/facenet_svm_classifier (1).pkl')  
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                face_crop = frame[max(0, y):y + h_box, max(0, x):x + w_box]

                if face_crop.size > 0:
                    label, confidence = classifier.predict(face_crop)

                    cv2.putText(frame, f"Class: {label}, {confidence*100:.2f}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                drawing.draw_detection(frame, detection)

        cv2.imshow('Face Classifier', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
