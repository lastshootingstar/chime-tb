import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

# Setup MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

input_folder = "C:/Users/drraj/Desktop/chime-tb/videos"
video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

for video_file in video_files:
    print(f"\nProcessing video: {video_file}")
    cap = cv2.VideoCapture(os.path.join(input_folder, video_file))

    prev_gray = None
    respiration_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape

            # Helper function to get pixel coords from landmarks
            def get_coords(idx):
                lm = results.pose_landmarks.landmark[idx]
                return int(lm.x * w), int(lm.y * h)

            # Get chest box corners from shoulders and hips
            l_shoulder = get_coords(11)
            r_shoulder = get_coords(12)
            l_hip = get_coords(23)
            r_hip = get_coords(24)

            top_left = (min(l_shoulder[0], r_shoulder[0]), min(l_shoulder[1], r_shoulder[1]))
            bottom_right = (max(l_hip[0], r_hip[0]), max(l_hip[1], r_hip[1]))

            # Draw chest box on frame
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Optical flow requires grayscale images
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                prev_chest = prev_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                curr_chest = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                flow = cv2.calcOpticalFlowFarneback(prev_chest, curr_chest,
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                vertical_flow = flow[..., 1]
                mean_vertical_movement = np.mean(vertical_flow)
                respiration_signal.append(mean_vertical_movement)

                # Print breathing movement
                print(f"Frame vertical movement: {mean_vertical_movement:.4f}")

            prev_gray = gray.copy()

        cv2.imshow("Chest Movement & Breathing Detection", frame)

        if cv2.waitKey(10) & 0xFF == 27:  # Press ESC to skip to next video
            break

    # After video processing, calculate respiratory rate from signal
    respiration_signal = np.array(respiration_signal)
    # Smooth signal by simple moving average
    smoothed = np.convolve(respiration_signal, np.ones(5)/5, mode='valid')
    peaks, _ = find_peaks(smoothed, distance=15)  # adjust distance for breathing rate sensitivity

    # Calculate breaths per minute (assuming 30 fps video)
    video_duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / 30
    breaths = len(peaks)
    respiratory_rate = (breaths / video_duration_sec) * 60 if video_duration_sec > 0 else 0

    print(f"Estimated respiratory rate: {respiratory_rate:.2f} breaths per minute")

    cap.release()

cv2.destroyAllWindows()
print("All videos processed.")
