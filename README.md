CSc 8830 – Computer Vision
Assignment 5–6: Motion Tracking & Real-Time Object Tracking

Student: Victor Solomon
Course: CSc 8830 – Computer Vision
Instructor: Dr. Ashwin Ashok
Institution: Georgia State University

📌 Overview

This repository contains all code, derivations, and demonstration material for Assignment 5–6 in CSc 8830: Computer Vision.
The assignment consists of:

Motion Tracking Theory

Derivation of the optical flow equation from first principles

Manual computation of motion estimates using two consecutive frames

Real-Time Object Tracking Implementations

(i) Marker-based tracking (Aruco / QR markers)

(ii) Markerless tracking using Lucas–Kanade optical flow

(iii) Segmentation-based tracking using SAM2 (segmentation generated offline and played back in real-time via NPZ masks)

A video demonstration of all three tracking systems, as well as a complete PDF report with derivations and example calculations, is included as part of the assignment submission.

📂 Repository Structure
assignment5-6/
│
├── src/
│   ├── marker_tracker.py          # ArUco marker-based tracking
│   ├── markerless_tracker.py      # Lucas–Kanade feature tracking
│   ├── sam2_tracker.py            # SAM2 segmentation-based tracking (offline masks)
│   ├── utils.py                   # Shared webcam/visualization helpers
│   └── __init__.py
│
├── data/
│   ├── frames/                    # Frames extracted from Problem 1 videos (for motion calc)
│   ├── videos/                    # Optional prerecorded videos for SAM2 demo
│   ├── sam2_masks.npz             # Offline segmentation masks for SAM2 tracking
│   └── markers/                   # Printable ArUco markers
│
├── report/
│   ├── assignment5-6.pdf          # Full derivations + manual computation + results
│   └── figures/                   # Gradients, flow matrices, screenshots
│
├── results/
│   ├── demo_output/               # Screenshots, tracked sequences
│   └── demo_video.mp4             # Recorded demonstration
│
├── requirements.txt               # Python dependencies
└── README.md                      # You are here

🔧 Installation & Setup
1. Create and activate environment
conda create -n cv56 python=3.10 -y
conda activate cv56
pip install -r requirements.txt

2. Required Python Packages
opencv-python
numpy
matplotlib
torch               # required only for SAM2 (if generating new masks)


If using SAM2 to generate masks offline, SAM2 must be installed separately from Meta’s repository.

▶️ Running Each Tracking System
1. Marker-Based Tracking (Aruco)
python src/marker_tracker.py


This script:

Opens your webcam

Detects ArUco markers

Tracks their 2D position in real time

Draws corners, ID numbers, and center points

You may print markers from data/markers/.

2. Markerless Tracking (Lucas–Kanade Optical Flow)
python src/markerless_tracker.py


This script:

Detects Shi-Tomasi corners

Tracks features over time using pyramidal Lucas–Kanade

Draws motion trails and reinitializes when tracking is lost

Press r to reset feature detection. Press q to quit.

3. SAM2 Segmentation-Based Tracking

Prerequisite: Precomputed SAM2 masks stored in data/sam2_masks.npz.

Run:

python src/sam2_tracker.py


This script:

Loads a prerecorded video and segmentation masks

Overlays segmentation on each frame

Computes bounding boxes of segmented objects

Produces a real-time visualization

This satisfies the requirement to use SAM2 segmentation offline while demonstrating it online.

✏️ Part (a) — Motion Tracking Equation & Manual Computation

Section 1 of the PDF contains:

Full derivation of the brightness constancy assumption

Taylor expansion leading to

𝐼
𝑥
𝑢
+
𝐼
𝑦
𝑣
+
𝐼
𝑡
=
0
I
x
	​

u+I
y
	​

v+I
t
	​

=0

Matrix formulation for a local Lucas–Kanade window

Manual calculations using two actual consecutive frames

Computation of 
𝐼
𝑥
,
𝐼
𝑦
,
𝐼
𝑡
I
x
	​

,I
y
	​

,I
t
	​


Construction of the matrix 
𝐴
A

Solving 
(
𝐴
⊤
𝐴
)
𝑤
=
𝐴
⊤
𝑏
(A
⊤
A)w=A
⊤
b

Final numerical motion estimate 
(
𝑢
,
𝑣
)
(u,v)

All steps are shown in detail as required by the assignment.

🎥 Demonstration Video

The demonstration video (demo_video.mp4) includes:

Marker-based tracking

Markerless optical flow tracking

SAM2 segmentation-based tracking

Each system is shown operating on live webcam input or prerecorded video.

The video is uploaded separately to Google Classroom as required.

📑 Assignment Report

The final PDF includes:

Derivation of the motion equation

Manual computation of motion for two frames

Explanation of each tracking method

Screenshots and analysis

References (per assignment requirement)

The PDF is located in:

report/assignment5-6.pdf

