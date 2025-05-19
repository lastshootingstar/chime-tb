import joblib
import pandas as pd

model = joblib.load("../models/tb_detector.pkl")
new_data = pd.read_csv("../data/new_sample.csv")  # Feature data
pred = model.predict(new_data)
print("Prediction:", pred[0])
# The prediction will be either 0 (no TB) or 1 (TB)
# Note: Make sure to adjust the paths and filenames according to your setup and requirements
# You can also use argparse to handle command line arguments
