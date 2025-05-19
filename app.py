import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(layout="wide", page_title="Chest Wall Movement Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Dashboard Menu")
option = st.sidebar.radio("Navigate", ["Live Webcam", "Upload Video", "About"])

if option == "About":
    st.sidebar.markdown("""
    **Chest Wall Movement Analysis Dashboard**

    Developed with MediaPipe & OpenCV  
    - Real-time respiratory rate detection  
    - Professional dashboard UI  
    - For non-technical users

    Created by Dr. Raj  
    """)

def analyze_frame(frame, prev_l_shoulder_y, prev_r_shoulder_y):
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

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw chest bounding box
        top_left = (min(l_shoulder[0], r_shoulder[0]), min(l_shoulder[1], r_shoulder[1]))
        bottom_right = (max(l_hip[0], r_hip[0]), max(l_hip[1], r_hip[1]))

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Calculate vertical movement of shoulders
        if prev_l_shoulder_y is not None and prev_r_shoulder_y is not None:
            movement_l = l_shoulder[1] - prev_l_shoulder_y
            movement_r = r_shoulder[1] - prev_r_shoulder_y
            movement = (movement_l + movement_r) / 2

        prev_l_shoulder_y = l_shoulder[1]
        prev_r_shoulder_y = r_shoulder[1]

    return frame, movement, prev_l_shoulder_y, prev_r_shoulder_y

if option == "Live Webcam":
    st.title("Live Chest Wall Movement Analysis")

    run = st.checkbox('Start Webcam Analysis')

    # Placeholders for video and analysis text
    video_placeholder = st.empty()
    analysis_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        prev_l_shoulder_y, prev_r_shoulder_y = None, None

        movements = []
        start_time = time.time()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to capture video")
                break

            frame, movement, prev_l_shoulder_y, prev_r_shoulder_y = analyze_frame(frame, prev_l_shoulder_y, prev_r_shoulder_y)

            # Store movement data for respiratory rate calculation
            movements.append(movement)

            # Calculate respiratory rate every 30 seconds (example)
            elapsed = time.time() - start_time
            if elapsed > 30:
                # Simple peak detection for breaths
                peaks = 0
                for i in range(1, len(movements)-1):
                    if movements[i-1] < movements[i] > movements[i+1] and movements[i] > 1:  # threshold
                        peaks += 1
                respiratory_rate = (peaks / elapsed) * 60
                movements = []
                start_time = time.time()
            else:
                respiratory_rate = 0

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Feed")

            # Display running analysis on the right side
            with analysis_placeholder.container():
                st.markdown("### Running Analysis")
                st.write(f"**Chest Wall Vertical Movement:** {movement:.2f} pixels")
                if respiratory_rate > 0:
                    st.write(f"**Estimated Respiratory Rate:** {respiratory_rate:.1f} breaths/min")
                else:
                    st.write("Estimating respiratory rate...")

            # Allow stop checkbox to break loop
            run = st.checkbox('Start Webcam Analysis', value=True)

        cap.release()
        cv2.destroyAllWindows()

elif option == "Upload Video":
    st.title("Upload Chest Wall Video")

    uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        prev_l_shoulder_y, prev_r_shoulder_y = None, None
        movements = []
        start_time = time.time()

        video_placeholder = st.empty()
        analysis_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, movement, prev_l_shoulder_y, prev_r_shoulder_y = analyze_frame(frame, prev_l_shoulder_y, prev_r_shoulder_y)
            movements.append(movement)

            elapsed = time.time() - start_time
            if elapsed > 30:
                peaks = 0
                for i in range(1, len(movements)-1):
                    if movements[i-1] < movements[i] > movements[i+1] and movements[i] > 1:
                        peaks += 1
                respiratory_rate = (peaks / elapsed) * 60
                movements = []
                start_time = time.time()
            else:
                respiratory_rate = 0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", caption="Uploaded Video")

            with analysis_placeholder.container():
                st.markdown("### Running Analysis")
                st.write(f"**Chest Wall Vertical Movement:** {movement:.2f} pixels")
                if respiratory_rate > 0:
                    st.write(f"**Estimated Respiratory Rate:** {respiratory_rate:.1f} breaths/min")
                else:
                    st.write("Estimating respiratory rate...")

        cap.release()
        cv2.destroyAllWindows()
