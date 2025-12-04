import streamlit as st
import subprocess
import sys
import os

# =========================================================
# PATH INITIALIZATION
# =========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="CSc 8830 – Assignment 5 Object Tracking",
    layout="wide",
    page_icon=None   # Removed 🎯 icon
)

# Reusable function
def run_script(command_list):
    try:
        st.info(f"Running: {' '.join(command_list)}")
        subprocess.Popen([sys.executable] + command_list, cwd=PROJECT_ROOT)
        st.success("Tracker window started. Close it or press 'q' to stop tracking.")
    except Exception as e:
        st.error(f"Error: {e}")


# =========================================================
# PAGE HEADER
# =========================================================

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 25px 0;">
        <h1 style="font-size:40px; margin-bottom:5px;">CSc 8830 – Assignment 5</h1>
        <h3 style="margin-top:0;">Object Tracking Demonstration – ArUco • KLT • SAM2</h3>
        <p style="color:#888; font-size:16px;">Georgia State University • Computer Vision • Fall 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# =========================================================
# SIDEBAR — GENERAL INSTRUCTIONS
# =========================================================

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    **Welcome to the Assignment 5 demo interface.**

    This app lets you run **three object tracking methods**:

    1. **ArUco Marker Tracking**  
       Requires printed ArUco markers placed on the object.

    2. **KLT Markerless Tracking**  
       You will select an ROI at the start.

    3. **SAM2 Segmentation Tracking**  
       Uses precomputed `.npz` masks for segmentation-driven tracking.

    ### How to Use:
    - Click a tab.
    - Read method instructions.
    - Click **Start Tracker**.
    - A window will open showing the real-time tracker.
    - Press **q** inside the window to stop tracking.
    """
)

st.sidebar.write("---")
st.sidebar.markdown(
    " CSc 8830 – Computer Vision"
)

# =========================================================
# TABS FOR THREE METHODS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    ["ArUco Marker Tracker", "KLT (Markerless)", "SAM2-Based Tracker"]
)

# =========================================================
# TAB 1 — ARUCO
# =========================================================

with tab1:
    st.markdown("## ArUco Marker-Based Tracking")
    st.markdown(
        """
        **Description:**  
        This method detects and tracks ArUco markers (DICT_4X4_50) placed on the object.  
        The system identifies:
        - Marker ID  
        - Corner boundaries  
        - Center position  

        **Requirements:**  
        - Print the provided ArUco marker.  
        - Place one marker centered on the object.  
        - Use good lighting.

        **Output:**  
        A real-time window with bounding boxes and marker IDs.
        """
    )

    st.info("Click the button below to launch the ArUco tracker in a new window.")

    if st.button("Start ArUco Tracker", key="aruco_btn", type="primary"):
        aruco_script = os.path.join(PROJECT_ROOT, "src", "aruco_tracker.py")
        video_path   = os.path.join(PROJECT_ROOT, "data", "videos", "aruco_demo.mp4")
        run_script([aruco_script, "--video", video_path])


# =========================================================
# TAB 2 — KLT
# =========================================================

with tab2:
    st.markdown("## KLT (Markerless) Tracker")
    st.markdown(
        """
        **Description:**  
        This method uses the classic **Lucas–Kanade optical flow** algorithm.  
        It tracks feature points inside a Region of Interest (ROI) that *you* select.

        **Instructions:**  
        - When the window opens, **drag to select the object**.  
        - Press **ENTER** to confirm the ROI.  
        - The tracker will follow features and update the bounding box.

        **Output:**  
        A real-time tracker window showing:
        - KLT features (green)  
        - Tracked bounding box (red)
        """
    )

    st.info("Click below to launch the KLT markerless tracker.")

    if st.button("Start KLT Tracker", key="klt_btn", type="primary"):
        klt_script = os.path.join(PROJECT_ROOT, "src", "klt_tracker.py")
        video_path = os.path.join(PROJECT_ROOT, "data", "videos", "klt_demo.mp4")
        run_script([klt_script, "--video", video_path])


# =========================================================
# TAB 3 — SAM2 SEGMENTATION
# =========================================================

with tab3:
    st.markdown("## SAM2 Segmentation-Based Tracker")
    st.markdown(
        """
        **Description:**  
        This method uses **precomputed segmentation masks** (NPZ format) 
        to simulate the behavior of SAM2 semantic segmentation tracking.

        **Instructions:**  
        - You do not need to select ROI.  
        - The mask for each frame is loaded and used to draw segmentation + bounding box.

        **Output:**  
        A real-time tracker showing:
        - Mask overlay (green transparent)
        - Bounding box over the segmented object
        """
    )

    st.info("Click below to launch the SAM2 tracker.")

    if st.button("Start SAM2 Tracker", key="sam2_btn", type="primary"):
        sam2_script = os.path.join(PROJECT_ROOT, "src", "sam2_tracker.py")
        video_path  = os.path.join(PROJECT_ROOT, "data", "videos", "klt_demo.mp4")
        mask_path   = os.path.join(PROJECT_ROOT, "data", "sam2_masks", "klt_kltmasks.npz")
        run_script([sam2_script, "--video", video_path, "--masks", mask_path])
