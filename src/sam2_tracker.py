#!/usr/bin/env python3
"""
sam2_tracker.py

Real-time-like tracker that uses precomputed segmentation masks
(simulating SAM2 output) stored in an .npz file.

- Video is read frame-by-frame.
- For each frame, a mask is loaded from the masks array.
- A bounding box and overlay are drawn on the frame.

Usage:
    python src/sam2_tracker.py --video data/videos/klt_demo.mp4 --masks data/sam2_masks/klt_demo_masks.npz

Press 'q' to quit.
"""

import cv2
import numpy as np
import argparse


def ensure_upright(frame):
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 segmentation-based tracker")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--masks", required=True, help="Path to npz file with masks")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load masks from npz
    data = np.load(args.masks)
    if "masks" not in data:
        print("[ERROR] NPZ file must contain array 'masks'.")
        return

    masks = data["masks"]  # (N, H, W), uint8 or bool
    N_masks = masks.shape[0]
    print(f"[INFO] Loaded masks with shape: {masks.shape}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    # Get first frame to size the window
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        return

    frame = ensure_upright(frame)
    H, W = frame.shape[:2]

    cv2.namedWindow("SAM2 Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SAM2 Tracker", W, H)

    print("[INFO] Starting SAM2-based tracking. Press 'q' to quit.")

    frame_idx = 0
    last_bbox = None

    # Rewind video to first frame (we already read one)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        frame = ensure_upright(frame)

        # Select corresponding mask index (clamp if video longer than masks)
        if frame_idx < N_masks:
            mask = masks[frame_idx]
        else:
            # If we have no mask for this frame, reuse last
            if last_bbox is None:
                mask = np.zeros((H, W), dtype=np.uint8)
            else:
                mask = np.zeros((H, W), dtype=np.uint8)
                x1, y1, x2, y2 = last_bbox
                mask[y1:y2, x1:x2] = 1

        # Ensure mask shape matches frame after rotation
        if mask.shape != (H, W):
            print(f"[WARN] Mask shape {mask.shape} does not match frame {H,W}. Resizing mask.")
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        # Binary mask in {0,1}
        mask_bin = (mask > 0).astype(np.uint8)

        # Compute bounding box from mask, if any pixels are foreground
        ys, xs = np.where(mask_bin == 1)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            last_bbox = (x_min, y_min, x_max, y_max)
        # If no nonzero pixels and we have a last_bbox, keep drawing that

        # Create colored overlay for mask
        overlay = frame.copy()
        overlay[mask_bin == 1] = (0, 255, 0)  # green overlay for mask

        # Blend overlay with original frame
        alpha = 0.4
        vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw bounding box if available
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                "SAM2 object",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("SAM2 Tracker", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed. Exiting.")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
