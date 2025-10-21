import os
import pandas as pd

# --- Load dataset ---
df = pd.read_csv("data/raw_data.csv")

# Identify rows where screen orientation is not 1 (invalid)
mask = df["screen_Orientation"] != 1

# --- Remove invalid frame files ---
for frame_path in df.loc[mask, "frame_path"]:
    if os.path.exists(frame_path):
        try:
            os.remove(frame_path)
        except Exception as e:
            print(f"Error removing {frame_path}: {e}")

# Keep only valid rows (orientation == 1)
df_clean = df.loc[~mask].copy()
df_clean.to_csv("data_cleaned.csv", index=False)

# --- Define which columns to ignore during validation ---
exclude_columns = ["dot_XCam", "dot_YCam", "dot_XPts", "dot_YPts"]

def is_invalid(value, column):

    # Values <= 0 are considered invalid, unless in exclude_columns.
    if column in exclude_columns:
        return False
    if isinstance(value, (int, float)):
        return value <= 0
    try:
        return float(value) <= 0
    except Exception:
        return False

# --- Check for invalid numeric data and remove related frames ---
rows_to_drop = []

for index, row in df_clean.iterrows():
    for col in df_clean.columns:
        if is_invalid(row[col], col):
            frame_path = row.get("frame_path")

            if isinstance(frame_path, str) and os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                    print(f"Removed file: {frame_path}")
                except Exception as e:
                    print(f"Error removing file {frame_path}: {e}")

            rows_to_drop.append(index)
            break 

# Drop invalid rows and save cleaned dataset
df_clean.drop(index=rows_to_drop, inplace=True)
df_clean.to_csv("data/data_cleaned.csv", index=False)
