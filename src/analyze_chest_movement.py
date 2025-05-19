import os
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

input_folder = "C:/Users/drraj/Desktop/chime-tb/videos"  # Folder with sample videos
video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

for video_file in video_files:
    print(f"\nProcessing: {video_file}")
    cap = cv2.VideoCapture(os.path.join(input_folder, video_file))

    prev_l_shoulder_y = None
    prev_r_shoulder_y = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = frame.shape

            def get_coords(idx):
                lm = results.pose_landmarks.landmark[idx]
                return int(lm.x * w), int(lm.y * h)

            l_shoulder = get_coords(11)
            r_shoulder = get_coords(12)
            l_hip = get_coords(23)
            r_hip = get_coords(24)

            # Draw chest bounding box
            top_left = (min(l_shoulder[0], r_shoulder[0]), min(l_shoulder[1], r_shoulder[1]))
            bottom_right = (max(l_hip[0], r_hip[0]), max(l_hip[1], r_hip[1]))

            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Calculate vertical chest movement
            if prev_l_shoulder_y is not None and prev_r_shoulder_y is not None:
                movement_l_shoulder = l_shoulder[1] - prev_l_shoulder_y
                movement_r_shoulder = r_shoulder[1] - prev_r_shoulder_y
                avg_movement = (movement_l_shoulder + movement_r_shoulder) / 2
                print(f"Chest wall vertical movement: {avg_movement:.2f}")

            prev_l_shoulder_y = l_shoulder[1]
            prev_r_shoulder_y = r_shoulder[1]

        cv2.imshow("Analyzing Chest Wall Movement", frame)

        # ESC key to skip to next video
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()

cv2.destroyAllWindows()
print("All videos processed.")
