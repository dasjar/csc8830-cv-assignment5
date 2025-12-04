#!/usr/bin/env python3
"""
klt_tracker.py — FINAL CRASH-PROOF VERSION
- User selects ROI
- ROI always visible
- KLT optical flow updates ROI only when valid
- Never crashes even with malformed LK results
- Portrait videos auto-rotated
"""

import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Markerless Lucas-Kanade tracker")
    parser.add_argument("--video", required=True, help="Path to video file")
    return parser.parse_args()


def ensure_upright(frame):
    """Rotate portrait video (H > W) into landscape (W >= H)."""
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Empty video.")
        return

    frame = ensure_upright(frame)
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = frame.shape[:2]

    # ROI selection
    cv2.namedWindow("Select ROI (KLT)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI (KLT)", W, H)

    print("[INFO] Select ROI then press ENTER.")
    roi = cv2.selectROI("Select ROI (KLT)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI (KLT)")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[ERROR] Invalid ROI.")
        return

    # ROI bounding box, always drawn
    roi_box = np.array([x, y, x + w, y + h], dtype=np.float32)

    # Create feature mask
    mask = np.zeros_like(old_gray)
    mask[y:y + h, x:x + w] = 255

    # Robust feature detection
    feature_params = dict(
        maxCorners=400,
        qualityLevel=0.001,
        minDistance=4,
        blockSize=5
    )

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
    if p0 is None:
        p0 = np.zeros((0, 1, 2), dtype=np.float32)

    print(f"[INFO] Initial features detected: {len(p0)}")

    # Lucas–Kanade optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    cv2.namedWindow("KLT Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("KLT Tracker", W, H)

    print("[INFO] Tracking started. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = ensure_upright(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === Compute optical flow safely ===
        if len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            # Handle p1=None or malformed
            if p1 is None or st is None:
                good_new = np.zeros((0, 2), dtype=np.float32)
            else:
                st = st.reshape(-1)
                valid = st == 1
                good_new = p1[valid] if np.any(valid) else np.zeros((0, 2), dtype=np.float32)
        else:
            good_new = np.zeros((0, 2), dtype=np.float32)

        # === Normalize shape to (N,2) ALWAYS ===
        good_new = np.array(good_new, dtype=np.float32).reshape(-1, 2)

        # === Update ROI ONLY if enough points exist ===
        if good_new.shape[0] >= 4:
            x_min = float(np.min(good_new[:, 0]))
            y_min = float(np.min(good_new[:, 1]))
            x_max = float(np.max(good_new[:, 0]))
            y_max = float(np.max(good_new[:, 1]))
            roi_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # === ALWAYS DRAW ROI, even if 0 points ===
        x1, y1, x2, y2 = roi_box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # === Draw tracked points ===
        for (cx, cy) in good_new:
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        # Show frame
        cv2.imshow("KLT Tracker", frame)

        # Update for next iteration
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if good_new.shape[0] > 0 else np.zeros((0, 1, 2), dtype=np.float32)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
