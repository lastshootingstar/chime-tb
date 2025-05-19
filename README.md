# CHIME-TB

**CHIME-TB** (Chest wall Imaging and Motion Estimation for Tuberculosis) is a video-based tool using **smartphone-recorded chest motion** to assist in early detection of **pulmonary tuberculosis (TB)** using AI/ML.

This project is a proof-of-concept to explore whether subtle thoracic motion abnormalities during quiet breathing or deep breaths can help screen for TB-related lung dysfunction.

---

## 📦 Folder Structure

CHIME-TB/
├── data/ # Raw and processed keypoint & feature data

├── videos/ # Chest motion videos (.mp4)
├── src/ # Source code for processing and modeling
├── models/ # Trained ML models
├── notebooks/ # Jupyter notebooks for exploration
├── results/ # Metrics, graphs, evaluation
├── README.md
├── requirements.txt
├── .gitignore


---

## 🛠️ Installation

```bash
git clone https://github.com/lastshootingstar/chime-tb/
cd CHIME-TB
pip install -r requirements.txt

📹 How to Use

Place your chest video in the videos/ folder

Run extract_keypoints.py to detect landmarks using MediaPipe

Run extract_features.py to get breathing features

Train a classifier with train_model.py

Predict new samples using predict.py

🤖 Model

Pose Estimation: MediaPipe Pose

ML Models: Random Forest, Logistic Regression (scikit-learn)

Language: Python 3.10+

Visualization: Matplotlib, Seaborn

Input: Chest landmark motion over time

Output: TB or Non-TB (binary label)

🧪 Sample Data

(You can optionally link to sample videos or dummy data)

📜 License

MIT License (or mention if for academic use only)

🙏 Acknowledgements

MediaPipe by Google

OpenAI ChatGPT for technical and writing support


