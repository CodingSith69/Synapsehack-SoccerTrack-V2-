import pandas as pd
import numpy as np
import cv2
import os

csv_path = 'data/interim/117093/117093_detections.csv'

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit()

# Load CSV WITHOUT header, and manually name the columns
df = pd.read_csv(csv_path, header=None)
df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x_null', 'y_null', 'z_null', 'class_name']

print("Data Loaded. Sample BBox:", df.iloc[0][['bb_left', 'bb_top']].values)

# --- THE HOMOGRAPHY MATRIX ---
H = np.array([
    [ 1.5e-03, -2.1e-04, -1.2e+00],
    [ 4.5e-05,  1.1e-03, -4.5e-01],
    [ 2.1e-07,  5.2e-06,  1.0e-03]
])

def transform_coords(df, H):
    # Calculate foot position (Bottom Center of BBox)
    foot_x = df['bb_left'] + df['bb_width'] / 2
    foot_y = df['bb_top'] + df['bb_height']
    
    pts = np.array([foot_x, foot_y]).T
    pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
    
    # Apply Homography
    transformed = cv2.perspectiveTransform(pts_reshaped, H)
    transformed = transformed.reshape(-1, 2)
    
    df['x_metres'] = transformed[:, 0]
    df['y_metres'] = transformed[:, 1]
    return df

try:
    print("Projecting players onto 2D Grassroot Pitch...")
    df_pitch = transform_coords(df, H)
    
    output_path = 'data/interim/117093/117093_pitch_plane_coordinates.csv'
    df_pitch.to_csv(output_path, index=False)
    print(f"Success! Tactical data saved to {output_path}")
except Exception as e:
    print(f"Failed: {e}")