import streamlit as st

st.set_page_config(
    page_title="Assignment 5 – Demo Video",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 25px 0;">
        <h1 style="font-size:40px; margin-bottom:5px;">CSc 8830 – Assignment 5</h1>
        <h3 style="margin-top:0;">Demonstration Video of Object Tracking Web App</h3>
        <p style="color:#888; font-size:16px;">Georgia State University • Computer Vision • Fall 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

st.header("📽️ Live App Demonstration – Recorded Video")

st.markdown(
    """
    **Description:**  
    This video is a recorded demonstration showing how the full Streamlit web application
    (`app.py`) works on a local machine, covering all three object tracking methods:

    - ArUco Marker Tracking  
    - KLT (Markerless) Tracking  
    - SAM2 Segmentation-Based Tracking  

    The recording shows:
    - The Streamlit interface layout  
    - How each tab launches a real-time tracker  
    - How OpenCV windows are used for visualization  
    - Expected behavior during tracking  

    This is included so that instructors and reviewers can see the application's
    behavior even though OpenCV GUI windows cannot run on cloud environments like Streamlit Cloud.
    """
)

# ✅ Working GitHub Releases streaming URL
VIDEO_URL = "https://github.com/dasjar/csc8830-cv-assignment5/releases/download/demo/klt_demo_output.mp4"

st.video(VIDEO_URL)

st.write("---")

st.markdown(
    """
    ### 📌 Notes
    - The real-time tracking is fully implemented locally.
    - Cloud environments cannot render OpenCV GUI windows, so this video serves as the official live demo.
    - To run the full application locally, execute:

      ```
      streamlit run app/app.py
      ```
    """
)
