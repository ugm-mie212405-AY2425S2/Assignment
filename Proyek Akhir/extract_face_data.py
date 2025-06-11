import os
import face_recognition
import pickle
import cv2 # Import OpenCV for robust image loading and format conversion

# --- Configuration ---
dataset_dir = "dataset_augmented"
output_file = "face_data.pkl"

# Structure for storing face encodings and metadata
data = {
    "encodings": [],
    "labels": [], # e.g., "mahasiswa", "dosen"
    "names": []   
}

print(f"[START] Starting face encoding process from '{dataset_dir}'...")

# --- Process Dataset ---
# Loop through main roles (mahasiswa, dosen)
for role in ["mahasiswa", "dosen"]:
    role_path = os.path.join(dataset_dir, role)
    print(f"\n[INFO] Checking role path: {role_path}")

    # Check if the role directory exists
    if not os.path.exists(role_path):
        print(f"[WARNING] Role directory not found, skipping: {role_path}")
        continue

    # Loop through each person's folder within the role
    for person_name in os.listdir(role_path):
        person_dir = os.path.join(role_path, person_name)

        # Skip if it's not a directory (e.g., a hidden file)
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing person: {os.path.join(role, person_name)}")

        # Loop through each image file in the person's folder
        for img_name in os.listdir(person_dir):
            # Check for common image file extensions
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(person_dir, img_name)

                # --- Image Loading with OpenCV ---
                # Load image using OpenCV (loads as BGR by default)
                bgr_image = cv2.imread(img_path)

                # Check if image loading was successful
                if bgr_image is None:
                    print(f"[ERROR] Could not load image (might be corrupted or invalid): {img_path}")
                    continue

                # Convert BGR image to RGB (required by face_recognition)
                # Ensure it's a 3-channel image before conversion
                if len(bgr_image.shape) == 3 and bgr_image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                elif len(bgr_image.shape) == 2: # Handle grayscale images by converting to RGB
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2RGB)
                else:
                    print(f"[WARNING] Unsupported channel count ({bgr_image.shape}) for: {img_path}. Skipping.")
                    continue

                # --- Face Encoding ---
                try:
                    # Extract face encodings from the RGB image
                    encodings = face_recognition.face_encodings(rgb_image)

                    if encodings:
                        # Append the first detected face encoding
                        data["encodings"].append(encodings[0])
                        data["labels"].append(role)
                        data["names"].append(person_name)
                        print(f"  [SUCCESS] Encoded face from: {img_name}")
                    else:
                        print(f"  [WARNING] No face found in: {img_name}")
                except Exception as e:
                    print(f"  [ERROR] Failed to process {img_name} for {os.path.join(role, person_name)}. Error: {e}")

# --- Save Encoded Data ---
try:
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n[✅ DONE] Successfully saved {len(data['encodings'])} face encodings to '{output_file}'")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to save data to '{output_file}'. Error: {e}")