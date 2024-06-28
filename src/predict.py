# src/predict.py
import os
import sys
import numpy as np

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_data
from preprocessing import normalize_data, handle_missing_values, rescale_level_preds
from model import build_model

def predict(model, subject_path, min_level, max_level):
    try:
        # Load and preprocess data
        print(f"Loading data from {subject_path}...")
        fMRI_data, ppg_data, resp_data, _ = load_data(subject_path)
        if fMRI_data is None or ppg_data is None or resp_data is None:
            raise ValueError("Data loading failed. Check data format and paths.")

        print("Normalizing data...")
        fMRI_data, ppg_data, resp_data = normalize_data(fMRI_data, ppg_data, resp_data)
        fMRI_data = handle_missing_values(fMRI_data)
        ppg_data = handle_missing_values(ppg_data)
        resp_data = handle_missing_values(resp_data)

        fMRI_data = fMRI_data[..., np.newaxis]
        ppg_data = ppg_data[..., np.newaxis]
        resp_data = resp_data[..., np.newaxis]

        # Predict
        print("Making predictions...")
        predictions = model.predict([fMRI_data, ppg_data, resp_data])
        class_preds, level_preds = predictions[0], predictions[1]

        # Convert CLASS predictions to integers and adjust if necessary
        class_preds = np.argmax(class_preds, axis=1) - 1  # Assuming the softmax outputs for 3 classes: {0, 1, 2}

        # Reverse scaling for level predictions
        level_preds = rescale_level_preds(level_preds, min_level, max_level)

        # Clip level predictions to be within -4 to 4
        level_preds = np.clip(level_preds, -4, 4)

        return class_preds, level_preds
    except ValueError as e:
        print(f"ValueError: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

if __name__ == "__main__":
    test_subjects = ['P17', 'P18', 'P19', 'P20']
    model = build_model()
    model.load_weights('best_model.keras')
    min_level, max_level = 0.0, 1.0

    for subject in test_subjects:
        subject_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Test', subject)
        class_preds, level_preds = predict(model, subject_path, min_level, max_level)
        if class_preds is not None and level_preds is not None:
            print(f'Class Predictions for {subject}: {class_preds}, Level Predictions: {level_preds}')
        else:
            print(f"Prediction failed for {subject}.")
