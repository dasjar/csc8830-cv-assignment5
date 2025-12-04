#!/usr/bin/env python3
"""
Generate SAM2-style masks from KLT bounding boxes.

This script automatically:
- Runs the KLT tracker on a video
- Stores the bounding box per frame
- Converts bounding boxes to binary masks
- Saves masks to an NPZ file usable by sam2_tracker.py
"""

import cv2
import numpy as np
import argparse
from klt_tracker import ensure_upright  # reuse orientation fix


def parse_args():
    parser = argparse.ArgumentParser(description="Create masks from KLT bounding boxes")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--out", required=True, help="Output npz file")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    # read first frame for ROI selection
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Empty video.")
        return

    frame = ensure_upright(frame)
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = frame.shape[:2]

    # select ROI
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", W, H)
    print("[INFO] Select initial ROI for KLT.")
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[ERROR] Empty ROI.")
        return

    # initialize bounding box
    bbox = np.array([x, y, x+w, y+h], dtype=np.float32)

    # detect features inside ROI
    mask = np.zeros_like(old_gray)
    mask[y:y+h, x:x+w] = 255

    p0 = cv2.goodFeaturesToTrack(
        old_gray, mask=mask, maxCorners=400, qualityLevel=0.001,
        minDistance=4, blockSize=5
    )
    if p0 is None:
        p0 = np.zeros((0,1,2), dtype=np.float32)

    # LK params
    lk_params = dict(
        winSize=(21,21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # rewind video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    masks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = ensure_upright(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None and st is not None:
                st = st.reshape(-1)
                good_new = p1[st==1]
            else:
                good_new = np.zeros((0,2), dtype=np.float32)
        else:
            good_new = np.zeros((0,2), dtype=np.float32)

        good_new = np.array(good_new).reshape(-1,2)

        if len(good_new) >= 4:
            x_min = int(good_new[:,0].min())
            y_min = int(good_new[:,1].min())
            x_max = int(good_new[:,0].max())
            y_max = int(good_new[:,1].max())
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # build mask for this frame
        m = np.zeros((H,W), dtype=np.uint8)
        x1,y1,x2,y2 = bbox.astype(int)
        m[y1:y2, x1:x2] = 1
        masks.append(m)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2) if len(good_new)>0 else np.zeros((0,1,2),dtype=np.float32)

    masks = np.stack(masks, axis=0)
    print("[INFO] saving masks:", masks.shape)
    np.savez_compressed(args.out, masks=masks)
    print("[INFO] saved to", args.out)


if __name__ == "__main__":
    main()
