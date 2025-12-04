#!/usr/bin/env python3
"""
aruco_tracker.py

Real-time ArUco-based object tracker for CSc 8830 Assignment 5.

Usage:
    python src/aruco_tracker.py           # use default camera (index 0)
    python src/aruco_tracker.py --video path/to/video.mp4

Press 'q' to quit the viewer window.
"""

import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ArUco marker-based tracker")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file. If not set, use webcam (index 0)."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use if --video is not provided (default: 0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up video source: either a file or a camera
    if args.video is not None:
        print(f"[INFO] Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"[INFO] Opening camera index {args.camera}")
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    # Load ArUco dictionary and detector
    # We use the same DICT_4X4_50 as for marker generation
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("[INFO] Starting ArUco marker tracking. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or failed to grab frame.")
            break

        # Optionally resize for speed (uncomment if needed)
        # frame = cv2.resize(frame, (960, 540))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Draw marker borders and IDs on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # For each marker, compute its center and draw a small circle
            for marker_corners, marker_id in zip(corners, ids):
                # marker_corners shape: (1, 4, 2) -> 4 corners (x, y)
                pts = marker_corners[0]  # (4, 2)
                # Compute center as average of corners
                center_x = int(pts[:, 0].mean())
                center_y = int(pts[:, 1].mean())

                # Draw center
                cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)

                # Put text label with marker ID
                text = f"ID: {int(marker_id)}"
                cv2.putText(
                    frame,
                    text,
                    (center_x - 20, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # Show the frame
        cv2.imshow("ArUco Marker Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
