import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Create folder for output
output_folder = "../results/keypoints"
os.makedirs(output_folder, exist_ok=True)

def extract_keypoints_from_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)

    with open(output_csv, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        header = []
        for i in range(33):
            header += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
        csv_writer.writerow(header)

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                csv_writer.writerow(row)

                # Draw skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Draw chest wall region
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
            else:
                csv_writer.writerow([0]*132)

            # SHOW FRAME LIVE with overlay
            cv2.imshow('Chest Wall Tracking - Live', frame)

            # Exit on ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Keypoints extracted and saved to {output_csv}")

if __name__ == "__main__":
    test_video_path = "../videos/sample_chest_video.mp4"
    output_csv_path = os.path.join(output_folder, "sample_chest_video_keypoints.csv")
    extract_keypoints_from_video(test_video_path, output_csv_path)
