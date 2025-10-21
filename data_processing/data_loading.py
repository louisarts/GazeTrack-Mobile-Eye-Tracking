import os
import tarfile
import json
import pandas as pd

from src.config import RAW_DATA_PATH # Path to the raw GazeCapture data

# --- Setup directories ---
BASE_DIR = RAW_DATA_PATH 
SOURCE_DIR = os.path.join(BASE_DIR, "gazecapture")
TARGET_DIR = os.path.join(BASE_DIR, "gazecapture_data")

os.makedirs(TARGET_DIR, exist_ok=True)

# --- Extract .tar.gz files ---
tar_files = sorted(
    f for f in os.listdir(SOURCE_DIR)
    if f.endswith(".tar.gz") and not f.startswith("._")
)

extracted_count = 0
for file_name in tar_files:
    file_path = os.path.join(SOURCE_DIR, file_name)
    extract_path = os.path.join(TARGET_DIR, file_name.replace(".tar.gz", ""))

    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        os.remove(file_path)
        extracted_count += 1
    except tarfile.ReadError:
        print(f"ReadError: Could not extract {file_name}")
    except Exception as e:
        print(f"Error extracting {file_name}: {e}")

print(f"Done. Extracted {extracted_count} files.")

# --- Parse JSON and build DataFrame ---
rows = []
BASE_PATH = os.path.join(BASE_DIR, "gazecapture_data")

for i in range(100000):
    person_id = str(i).zfill(5)
    person_path = os.path.join(BASE_PATH, person_id, person_id)
    if not os.path.exists(person_path):
        continue

    try:
        with open(os.path.join(person_path, "appleFace.json")) as f:
            face_data = json.load(f)
        with open(os.path.join(person_path, "appleLeftEye.json")) as f:
            leye_data = json.load(f)
        with open(os.path.join(person_path, "appleRightEye.json")) as f:
            reye_data = json.load(f)
        with open(os.path.join(person_path, "faceGrid.json")) as f:
            grid_data = json.load(f)
        with open(os.path.join(person_path, "dotInfo.json")) as f:
            dot_data = json.load(f)
        with open(os.path.join(person_path, "screen.json")) as f:
            screen_data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON for person {person_id}: {e}")
        continue

    for j in range(10000):
        frame_id = f"{j:05d}.jpg"
        frame_path = os.path.join(person_path, "frames", frame_id)
        if not os.path.exists(frame_path):
            continue

        try:
            row = {
                "frame_path": frame_path,
                "face_box_X": face_data["X"][j],
                "face_box_Y": face_data["Y"][j],
                "face_box_W": face_data["W"][j],
                "face_box_H": face_data["H"][j],
                "left_eye_X": leye_data["X"][j],
                "left_eye_Y": leye_data["Y"][j],
                "left_eye_W": leye_data["W"][j],
                "left_eye_H": leye_data["H"][j],
                "right_eye_X": reye_data["X"][j],
                "right_eye_Y": reye_data["Y"][j],
                "right_eye_W": reye_data["W"][j],
                "right_eye_H": reye_data["H"][j],
                "dot_XPts": dot_data["XPts"][j],
                "dot_YPts": dot_data["YPts"][j],
                "dot_XCam": dot_data["XCam"][j],
                "dot_YCam": dot_data["YCam"][j],
                "screen_H": screen_data["H"][j],
                "screen_W": screen_data["W"][j],
                "screen_Orientation": screen_data["Orientation"][j],
            }
            rows.append(row)
        except (IndexError, KeyError) as e:
            print(f"Skipping frame {j} for person {person_id}: {e}")
            continue

# --- Save combined data ---
df = pd.DataFrame(rows)
df.to_csv("data/raw_data.csv", index=False)

print(f"DataFrame created with shape: {df.shape}")
