# test/test_data_loading_preprocessing.py
import os
import sys

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data
from src.preprocessing import normalize_data, handle_missing_values

# Set the path to the directory containing the data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Train')
subject_path = os.path.join(data_dir, 'P01')

# Load the data
fMRI_data, ppg_data, resp_data, labels = load_data(subject_path)
print("Original Shapes:")
print(f"fMRI_data: {fMRI_data.shape}, ppg_data: {ppg_data.shape}, resp_data: {resp_data.shape}, labels: {labels.shape}")

# Normalize the data
fMRI_data, ppg_data, resp_data = normalize_data(fMRI_data, ppg_data, resp_data)
print("Shapes after Normalization:")
print(f"fMRI_data: {fMRI_data.shape}, ppg_data: {ppg_data.shape}, resp_data: {resp_data.shape}")

# Handle missing values
ppg_data = handle_missing_values(ppg_data)
resp_data = handle_missing_values(resp_data)
print("Shapes after Handling Missing Values:")
print(f"fMRI_data: {fMRI_data.shape}, ppg_data: {ppg_data.shape}, resp_data: {resp_data.shape}")
