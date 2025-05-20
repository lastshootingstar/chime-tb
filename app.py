import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Page settings
st.set_page_config(layout="wide", page_title="Chest Wall Movement Analysis Dashboard")

# Sidebar
st.sidebar.title("Dashboard Menu")
option = st.sidebar.radio("Navigate", ["Live Webcam", "Upload Video", "About"])

if option == "About":
    st.sidebar.markdown("""
    **Chest Wall Movement Analysis Dashboard**

    Developed with MediaPipe & OpenCV  
    - Real-time respiratory rate detection  
    - Professional dashboard UI  

    Created by Dr. Raj  
    """)

# Frame analysis function
def analyze_frame(frame, prev_l_y, prev_r_y):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    movement = 0

    if results.pose_landmarks:
        h, w, _ = frame.shape

        def get_coords(idx):
            lm = results.pose_landmarks.landmark[idx]
            return int(lm.x * w), int(lm.y * h)

        l_shoulder = get_coords(11)
        r_shoulder = get_coords(12)
        l_hip = get_coords(23)
        r_hip = get_coords(24)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw chest region
        top_left = (min(l_shoulder[0], r_shoulder[0]), min(l_shoulder[1], r_shoulder[1]))
        bottom_right = (max(l_hip[0], r_hip[0]), max(l_hip[1], r_hip[1]))
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        if prev_l_y is not None and prev_r_y is not None:
            movement_l = l_shoulder[1] - prev_l_y
            movement_r = r_shoulder[1] - prev_r_y
            movement = (movement_l + movement_r) / 2

        prev_l_y, prev_r_y = l_shoulder[1], r_shoulder[1]

    return frame, movement, prev_l_y, prev_r_y

# Webcam mode
if option == "Live Webcam":
    st.title("Live Chest Wall Movement Analysis")

    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")

    video_placeholder = st.empty()
    analysis_placeholder = st.empty()

    if start_webcam and not stop_webcam:
        cap = cv2.VideoCapture(0)
        prev_l_y, prev_r_y = None, None
        movements = []
        start_time = time.time()
        respiratory_rate = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_webcam:
                break

            frame, movement, prev_l_y, prev_r_y = analyze_frame(frame, prev_l_y, prev_r_y)
            movements.append(movement)

            elapsed = time.time() - start_time
            if elapsed >= 60:
                peaks = sum(1 for i in range(1, len(movements)-1)
                            if movements[i-1] < movements[i] > movements[i+1] and movements[i] > 1)
                respiratory_rate = int((peaks / elapsed) * 60)
                movements = []
                start_time = time.time()

            # Show video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed")

            with analysis_placeholder.container():
                st.markdown("### Chest Wall Motion Analysis")
                st.write(f"**Vertical Movement:** {movement:.2f} pixels")
                st.markdown(f"<h3 style='color: green;'>Respiratory Rate: {respiratory_rate} breaths/min</h3>", unsafe_allow_html=True)

        cap.release()
        cv2.destroyAllWindows()

# Upload mode
elif option == "Upload Video":
    st.title("Upload Chest Wall Video")

    uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        prev_l_y, prev_r_y = None, None
        movements = []
        start_time = time.time()
        respiratory_rate = 0

        video_placeholder = st.empty()
        analysis_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, movement, prev_l_y, prev_r_y = analyze_frame(frame, prev_l_y, prev_r_y)
            movements.append(movement)

            elapsed = time.time() - start_time
            if elapsed >= 60:
                peaks = sum(1 for i in range(1, len(movements)-1)
                            if movements[i-1] < movements[i] > movements[i+1] and movements[i] > 1)
                respiratory_rate = int((peaks / elapsed) * 60)
                movements = []
                start_time = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", caption="Uploaded Video")

            with analysis_placeholder.container():
                st.markdown("### Chest Wall Motion Analysis")
                st.write(f"**Vertical Movement:** {movement:.2f} pixels")
                st.markdown(f"<h3 style='color: green;'>Respiratory Rate: {respiratory_rate} breaths/min</h3>", unsafe_allow_html=True)

        cap.release()
        cv2.destroyAllWindows()
