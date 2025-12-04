#!/usr/bin/env python3
"""
prepare_sam2_masks_from_roi.py

Offline mask generator to simulate SAM2-style segmentation output.
For each frame in the video:
- Rotate portrait -> landscape
- Use a user-selected ROI from the first frame
- Create a binary mask where the ROI region = 1, elsewhere 0

This produces an .npz file with:
- masks: (N, H, W) uint8 array
You can later replace the ROI-based mask generation with real SAM2 outputs.
"""

import cv2
import numpy as np
import argparse
import os


def ensure_upright(frame):
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SAM2 masks (ROI-based placeholder)")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--out",
        type=str,
        default="data/sam2_masks/masks.npz",
        help="Output npz path (default: data/sam2_masks/masks.npz)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    # Read first frame to define ROI and size
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        return

    frame = ensure_upright(frame)
    H, W = frame.shape[:2]

    # Let user select ROI on first frame
    cv2.namedWindow("Select ROI (SAM2 offline)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI (SAM2 offline)", W, H)
    print("[INFO] Select ROI for the object (SAM2 surrogate), then press ENTER.")
    roi = cv2.selectROI("Select ROI (SAM2 offline)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI (SAM2 offline)")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[ERROR] Empty ROI selected.")
        return

    # Pre-allocate masks list
    masks = []

    # First frame mask
    mask0 = np.zeros((H, W), dtype=np.uint8)
    mask0[y:y + h, x:x + w] = 1
    masks.append(mask0)

    # Process remaining frames
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = ensure_upright(frame)
        # For now, just reuse the same ROI location as mask
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y:y + h, x:x + w] = 1
        masks.append(mask)
        frame_idx += 1

    cap.release()

    masks = np.stack(masks, axis=0)  # (N, H, W)
    print(f"[INFO] Created masks array with shape: {masks.shape}")

    np.savez_compressed(args.out, masks=masks)
    print(f"[INFO] Saved masks to {args.out}")


if __name__ == "__main__":
    main()
