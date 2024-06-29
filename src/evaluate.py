import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import build_model
from src.data_loader import load_data
from src.preprocessing import normalize_data, handle_missing_values

def evaluate_model(model, val_data, val_labels, min_level, max_level):
    fMRI_val, ppg_val, resp_val = val_data
    class_val, level_val = val_labels

    predictions = model.predict([fMRI_val, ppg_val, resp_val])
    class_preds, level_preds = predictions[0], predictions[1]

    class_preds = np.argmax(class_preds, axis=1) - 1
    level_preds = level_preds * (max_level - min_level) + min_level
    level_preds = np.clip(level_preds, -4, 4)

    class_accuracy = accuracy_score(class_val, class_preds)
    level_rmse = np.sqrt(mean_squared_error(level_val, level_preds))

    print(f"Class Accuracy: {class_accuracy}")
    print(f"Level RMSE: {level_rmse}")
    return class_accuracy, level_rmse, class_preds, level_preds

def load_and_prepare_data(subject_path):
    fMRI_data, ppg_data, resp_data, labels = load_data(subject_path)
    fMRI_data, ppg_data, resp_data = normalize_data(fMRI_data, ppg_data, resp_data)
    fMRI_data = handle_missing_values(fMRI_data)
    ppg_data = handle_missing_values(ppg_data)
    resp_data = handle_missing_values(resp_data)

    fMRI_data = fMRI_data[..., np.newaxis]
    ppg_data = ppg_data[..., np.newaxis]
    resp_data = resp_data[..., np.newaxis]

    class_labels, level_labels = labels[:, 0], labels[:, 1]
    return (fMRI_data, ppg_data, resp_data), (class_labels, level_labels)

def load_test_data(test_dir):
    all_fMRI_data, all_ppg_data, all_resp_data, participant_trial_pairs = [], [], [], []

    for subject_folder in os.listdir(test_dir):
        subject_path = os.path.join(test_dir, subject_folder)
        if os.path.isdir(subject_path):
            fMRI_data, ppg_data, resp_data, _ = load_data(subject_path)
            if fMRI_data is not None and ppg_data is not None and resp_data is not None:
                all_fMRI_data.append(fMRI_data)
                all_ppg_data.append(ppg_data)
                all_resp_data.append(resp_data)
                for i in range(len(fMRI_data)):
                    participant_trial_pairs.append((subject_folder, i + 1))

    if not all_fMRI_data or not all_ppg_data or not all_resp_data:
        raise ValueError("Test data loading failed. Ensure the data format and paths are correct.")

    all_fMRI_data = np.concatenate(all_fMRI_data, axis=0)
    all_ppg_data = np.concatenate(all_ppg_data, axis=0)
    all_resp_data = np.concatenate(all_resp_data, axis=0)

    return all_fMRI_data, all_ppg_data, all_resp_data, participant_trial_pairs

def rescale_level_preds(level_preds, min_level, max_level):
    return level_preds * (max_level - min_level) + min_level

if __name__ == "__main__":
    val_subject_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Train', 'P16')
    val_data, val_labels = load_and_prepare_data(val_subject_path)

    model = build_model()
    model.load_weights('best_model.keras')
    min_level, max_level = -4.0, 4.0

    evaluate_model(model, val_data, val_labels, min_level, max_level)
