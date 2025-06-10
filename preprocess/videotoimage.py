import os
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def convert_video_to_30fps_and_crop_faces(video_path, output_dir, target_fps=30):
    output_path = os.path.join('dataset/per_frame/', output_dir)
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps
    total_target_frames = int(duration * target_fps)

    saved_frame = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        while cap.isOpened() and saved_frame < total_target_frames:
            time_sec = saved_frame / target_fps
            cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                for i, detection in enumerate(results.detections):
                    box = detection.location_data.relative_bounding_box
                    x_min = int(box.xmin * w)
                    y_min = int(box.ymin * h)
                    box_width = int(box.width * w)
                    box_height = int(box.height * h)

                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(x_min + box_width, w)
                    y_max = min(y_min + box_height, h)

                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size == 0:
                        continue

                    frame_filename = os.path.join(output_path, f"frame_{saved_frame:05d}.jpg")
                    cv2.imwrite(frame_filename, face_crop)
                    break  # Save only one face per frame

            saved_frame += 1

    cap.release()
    print(f"✅ Selesai! {saved_frame} frame dengan wajah disimpan di: {output_path}")

# Process all videos
input_folder = 'dataset/video'
for video_file in os.listdir(input_folder):
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    video_path = os.path.join(input_folder, video_file)
    output_folder = os.path.splitext(video_file)[0]
    convert_video_to_30fps_and_crop_faces(video_path, output_folder)
