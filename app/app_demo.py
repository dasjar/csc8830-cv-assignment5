import streamlit as st
import os

# ============================================
# Page Config
# ============================================

st.set_page_config(
    page_title="Assignment 5 – Tracking Demonstration Videos",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 25px 0;">
        <h1 style="font-size:40px; margin-bottom:5px;">CSc 8830 – Assignment 5</h1>
        <h3 style="margin-top:0;">Demonstration Videos for ArUco, KLT, and SAM2 Tracking</h3>
        <p style="color:#888; font-size:16px;">Georgia State University • Computer Vision • Fall 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "demo_videos"))


# ============================================
# Utility: Load a video safely
# ============================================

def load_video(filename):
    path = os.path.join(DEMO_DIR, filename)
    if not os.path.exists(path):
        st.warning(f"Video not found: {filename}")
        return None
    return path


# ============================================
# Tabs
# ============================================

tab1, tab2, tab3 = st.tabs([
    "ArUco Marker Tracking Demo",
    "KLT (Markerless) Tracking Demo",
    "SAM2-Based Tracking Demo"
])

# ============================================
# TAB 1 — ARUCO DEMO
# ============================================

with tab1:
    st.header("ArUco Marker-Based Tracking – Demonstration")

    st.markdown(
        """
        **What this video shows:**  
        - Detection of a **4×4 ArUco marker** placed on the object  
        - Automatic detection of marker ID  
        - Tracking of marker corners  
        - Overlay of bounding box + marker center  
        - Real-time orientation correction  

        This demonstration corresponds to **Part (ii).1** of the assignment.
        """
    )

    video_path = load_video("aruco_demo_output.mp4")
    if video_path:
        st.video(video_path)


# ============================================
# TAB 2 — KLT TRACKING DEMO
# ============================================

with tab2:
    st.header("Lucas–Kanade (Markerless) Tracking – Demonstration")

    st.markdown(
        """
        **What this video shows:**  
        - User selecting an ROI around the object  
        - Extraction and tracking of feature points  
        - Red bounding box updated dynamically  
        - Crash-proof tracking across frames  
        - Demonstration of markerless motion estimation  

        This corresponds to **Part (ii).2** of the assignment.
        """
    )

    video_path = load_video("klt_demo_output.mp4")
    if video_path:
        st.video(video_path)


# ============================================
# TAB 3 — SAM2 SEGMENTATION DEMO
# ============================================

with tab3:
    st.header("SAM2-Based Segmentation Tracking – Demonstration")

    st.markdown(
        """
        **What this video shows:**  
        - Precomputed segmentation masks applied frame-by-frame  
        - Dynamic bounding box overlay  
        - Mask visualization on top of the object  
        - Real-time tracking simulation using segmentation output  

        This corresponds to **Part (ii).3** of the assignment, where  
        **SAM2 segmentation is used to track the object**.
        """
    )

    video_path = load_video("sam2_demo_output.mp4")
    if video_path:
        st.video(video_path)
