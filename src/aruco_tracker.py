#!/usr/bin/env python3
"""
aruco_tracker.py – Real-time ArUco marker tracking.
"""

import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ArUco marker tracker")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. If not provided, webcam is used.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index to use if no video is provided.")
    return parser.parse_args()


def ensure_upright(frame):
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def main():
    args = parse_args()

    if args.video:
        print(f"[INFO] Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    # Load ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    print("[INFO] Tracking ArUco markers... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = ensure_upright(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw center points and ID text
            for pts, marker_id in zip(corners, ids):
                pts = pts[0]
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {int(marker_id)}",
                            (cx - 10, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        cv2.imshow("ArUco Marker Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
