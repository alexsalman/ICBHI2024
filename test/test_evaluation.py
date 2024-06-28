# test/test_evaluation.py
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data
from src.preprocessing import normalize_data, handle_missing_values, preprocess_labels, scale_level_labels
from src.model import build_model

def evaluate_model(model, X_val, y_val, min_level, max_level):
    predictions = model.predict(X_val)
    class_preds, level_preds = predictions[0], predictions[1]

    # Reverse scaling for level predictions
    level_preds = level_preds * (max_level - min_level) + min_level

    class_accuracy = accuracy_score(y_val[:, 0], np.argmax(class_preds, axis=1))
    level_mae = mean_absolute_error(y_val[:, 1], level_preds)

    return class_accuracy, level_mae

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Train')

    fMRI_data, ppg_data, resp_data, labels = [], [], [], []

    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        fMRI, ppg, resp, lbl = load_data(subject_path)
        fMRI_data.append(fMRI)
        ppg_data.append(ppg)
        resp_data.append(resp)
        labels.append(lbl)

    fMRI_data = np.concatenate(fMRI_data)
    ppg_data = np.concatenate(ppg_data)
    resp_data = np.concatenate(resp_data)
    labels = np.concatenate(labels)

    labels, valid_indices = preprocess_labels(labels)
    fMRI_data = fMRI_data[valid_indices]
    ppg_data = ppg_data[valid_indices]
    resp_data = resp_data[valid_indices]
    fMRI_data, ppg_data, resp_data = normalize_data(fMRI_data, ppg_data, resp_data)
    ppg_data = handle_missing_values(ppg_data)
    resp_data = handle_missing_values(resp_data)

    labels, min_level, max_level = scale_level_labels(labels)
    class_labels, level_labels = labels[:, 0], labels[:, 1]

    fMRI_data = fMRI_data[..., np.newaxis]
    ppg_data = ppg_data[..., np.newaxis]
    resp_data = resp_data[..., np.newaxis]

    fMRI_train, fMRI_val, ppg_train, ppg_val, resp_train, resp_val, y_train, y_val = train_test_split(
        fMRI_data, ppg_data, resp_data, labels, test_size=0.2, random_state=42)

    model = build_model()

    model.fit([fMRI_train, ppg_train, resp_train], [y_train[:, 0], y_train[:, 1]], validation_data=([fMRI_val, ppg_val, resp_val], [y_val[:, 0], y_val[:, 1]]), epochs=20, batch_size=32)

    class_accuracy, level_mae = evaluate_model(model, [fMRI_val, ppg_val, resp_val], y_val, min_level, max_level)
    print(f"Class Accuracy: {class_accuracy}, Level MAE: {level_mae}")
