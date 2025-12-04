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
    This video is a recorded demonstration showing how the full web application
    (`app.py`) works on the local machine.  
    It covers all three object tracking methods:

    - ArUco Marker Tracking  
    - KLT (Markerless) Tracking  
    - SAM2 Segmentation-Based Tracking  

    The demo shows:
    - The Streamlit interface  
    - How each tab launches the correct tracker  
    - What users should expect when running the application  
    - Real-time behavior of each tracking method  

    This video is included so that instructors and reviewers can see the
    application's behavior even if OpenCV windows cannot run in cloud
    environments.
    """
)

# Direct Google Drive streaming link
VIDEO_URL = "https://drive.google.com/uc?export=download&id=1S0txJPjGtpwschn70bVAH6P2FpvjzNDs"

st.video(VIDEO_URL)

st.write("---")

st.markdown(
    """
    ### 📌 Notes
    - The full real-time tracking system is implemented locally.
    - Cloud environments cannot render OpenCV GUI windows, so this recording
      serves as the official live demonstration.
    - For full interaction, run `app.py` locally using:

      ```
      streamlit run app/app.py
      ```

    """
)
