# test/test_prediction.py
import os
import sys
import numpy as np  # Import NumPy

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict import predict
from src.model import build_model
from src.data_loader import load_data

# Set the path to the directory containing the test data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Test')
subject_path = os.path.join(data_dir, 'P17')

# Build the model
model = build_model()

# Load the data for prediction
fMRI_data, ppg_data, resp_data, labels = load_data(subject_path)
fMRI_data = fMRI_data[..., np.newaxis]
ppg_data = ppg_data[..., np.newaxis]
resp_data = resp_data[..., np.newaxis]

# Predict on the test data
class_preds, level_preds = predict(model, subject_path)
print(f'Class Predictions: {class_preds}, Level Predictions: {level_preds}')
