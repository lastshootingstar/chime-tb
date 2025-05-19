import numpy as np
import pandas as pd

def compute_features(landmarks_sequence):
    y_coords = [frame[mp_index][1] for frame in landmarks_sequence]
    amplitude = max(y_coords) - min(y_coords)
    rate = len(y_coords) / 10  # dummy value (needs FPS)
    return {"amplitude": amplitude, "rate": rate}

if __name__ == "__main__":
    from extract_keypoints import extract_chest_landmarks
    data = extract_chest_landmarks("../videos/sample.mp4")
    features = compute_features(data)
    print(features)

