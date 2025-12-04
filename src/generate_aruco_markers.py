import cv2
import numpy as np
import os

# Output directory
out_dir = "data/markers"
os.makedirs(out_dir, exist_ok=True)

# Use a standard ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate markers with IDs 0–4
for marker_id in range(5):
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        400  # pixels
    )
    filename = os.path.join(out_dir, f"aruco_{marker_id}.png")
    cv2.imwrite(filename, marker_img)
    print(f"Saved {filename}")

print("\nDone! Open data/markers/ to see your markers.")
