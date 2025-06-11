import cv2
import face_recognition
import pickle
import numpy as np
import os
import csv
from datetime import datetime

# Load data and model
with open("face_data.pkl", 'rb') as f:
    data = pickle.load(f)
    
with open("svm_classifier.pkl", 'rb') as f:
    clf = pickle.load(f)
    
known_encodings = data["encodings"]
known_names = data["names"]

video_capture = cv2.VideoCapture(0)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

with open(log_file_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Timestamp", "Name", "Label"])
    
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_infos = []
            
            for face_encoding in face_encodings:
                label = "Unknown"
                name = "Unknown"
                
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < 0.45:
                        person_embedding = [face_encoding]
                        label = clf.predict(person_embedding)[0]
                        name = known_names[best_match_index]
                
                face_infos.append((label, name))
                writer.writerow([datetime.now().isoformat(), name, label])
                
        process_this_frame = not process_this_frame
        
        for(top, right, bottom, left), (label, name) in zip(face_locations, face_infos):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label.upper(), (left + 6, bottom - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, name, (left + 6, bottom - 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
        cv2.imshow("Real-time Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
video_capture.release()
cv2.destroyAllWindows()