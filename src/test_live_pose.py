import cv2
import mediapipe as mp

# Initialize pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

# Load webcam or video
cap = cv2.VideoCapture(0)  # <-- Use webcam (you can use path to video instead)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't read frame. Exiting.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw green chest wall overlay
        image_height, image_width, _ = frame.shape
        def get_coords(idx):
            lm = results.pose_landmarks.landmark[idx]
            return int(lm.x * image_width), int(lm.y * image_height)

        left_shoulder = get_coords(11)
        right_shoulder = get_coords(12)
        left_hip = get_coords(23)
        right_hip = get_coords(24)

        top_left = (min(left_shoulder[0], right_shoulder[0]), min(left_shoulder[1], right_shoulder[1]))
        bottom_right = (max(left_hip[0], right_hip[0]), max(left_hip[1], right_hip[1]))

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show the frame
    cv2.imshow("Live Pose + Chest Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
