import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# Load the coordinate data
csv_path = 'data/interim/117093/117093_pitch_plane_coordinates.csv'
df = pd.read_csv(csv_path)

# Video Settings
width, height = 1050, 680 # 10 pixels per meter
fps = 30
output_video = 'grassroots_talent_scout_view.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Generating Tactical Scout View...")

for frame_idx in tqdm(range(int(df['frame'].max() + 1))):
    # Create Green Pitch
    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34) # Forest Green
    
    # Draw Pitch Lines (Simplified)
    cv2.rectangle(pitch, (0, 0), (width, height), (255, 255, 255), 2) # Outer boundary
    cv2.line(pitch, (width//2, 0), (width//2, height), (255, 255, 255), 2) # Halfway line
    cv2.circle(pitch, (width//2, height//2), 91, (255, 255, 255), 2) # Center circle
    
    # Get players in this frame
    frame_data = df[df['frame'] == frame_idx]
    
    for _, player in frame_data.iterrows():
        # Scale meters to pixels (105m -> 1050px, 68m -> 680px)
        # Note: We add offsets if the homography isn't perfectly centered
        x = int(player['x_metres'] * 10)
        y = int(player['y_metres'] * 10)
        
        # Ensure dots stay on the pitch visual
        if 0 <= x < width and 0 <= y < height:
            # Draw Player Dot
            color = (0, 0, 255) if player['id'] % 2 == 0 else (255, 0, 0) # Red vs Blue teams
            cv2.circle(pitch, (x, y), 8, color, -1)
            cv2.putText(pitch, str(int(player['id'])), (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out.write(pitch)

out.release()
print(f"Success! Tactical Video saved as {output_video}")