import numpy as np
import json
import os

features = []
labels = []

label_map = {"healthy": 0, "TB": 1, "asthma": 2}

def encode_metadata(meta):
    return [
        int(meta["gender"] == "male"),
        float(meta["latitude"]),
        float(meta["longitude"]),
        int(meta["respiratory_condition"] == "True"),
        int(meta["fever_muscle_pain"] == "True"),
        int(meta["age"])
    ]

for file in os.listdir("metadata_json_folder"):
    if file.endswith(".json"):
        with open(os.path.join("metadata_json_folder", file)) as f:
            meta = json.load(f)
        status = meta["status"]
        if status not in label_map:
            continue

        # Load corresponding audio embedding (.npy)
        embedding_path = f"embeddings_folder/{file.replace('.json', '.npy')}"
        if not os.path.exists(embedding_path):
            continue

        embedding = np.load(embedding_path)
        metadata_vector = encode_metadata(meta)
        feature_vector = np.concatenate([embedding, metadata_vector])
        features.append(feature_vector)
        labels.append(label_map[status])

# Save dataset
np.save("features.npy", np.array(features))
np.save("labels.npy", np.array(labels))
print("✅ Saved features.npy and labels.npy")
