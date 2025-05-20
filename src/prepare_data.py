import os
import json
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch, soundfile as sf
from tqdm import tqdm

# Load audio model
proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_audio_embedding(path):
    audio, sr = sf.read(path)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb

# Store all samples
features = []
labels = []

# Your folder with CoughVID files
DATA_DIR = "data/coughvid"

for file in tqdm(os.listdir(DATA_DIR)):
    if file.endswith(".wav"):
        audio_path = os.path.join(DATA_DIR, file)
        meta_path = audio_path.replace(".wav", ".json")

        if not os.path.exists(meta_path):
            continue

        try:
            # Extract audio embedding
            audio_feat = extract_audio_embedding(audio_path)

            # Load metadata
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Filter unclear cases
            label = meta.get("status")
            if label not in ["healthy", "asthma", "COVID-19"]:  # or "TB" if you have
                continue

            # Simple metadata features
            age = float(meta.get("age", 30))
            gender = 1 if meta.get("gender") == "male" else 0
            fever = 1 if meta.get("fever_muscle_pain") == "True" else 0
            resp_cond = 1 if meta.get("respiratory_condition") == "True" else 0

            meta_feat = np.array([age, gender, fever, resp_cond])

            full_feat = np.concatenate([audio_feat, meta_feat])
            features.append(full_feat)
            labels.append(label)
        except Exception as e:
            print("Error:", e)
            continue

# Save features and labels
np.save("features.npy", np.array(features))
np.save("labels.npy", np.array(labels))
print("✅ Saved features.npy and labels.npy")
