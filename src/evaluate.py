# src/evaluate.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from data_loader import load_data
from model import build_model
from preprocessing import handle_missing_values

def evaluate_model(model, X_test, y_test):
    y_test = np.array(y_test)
    predictions = model.predict(X_test)
    class_preds, level_preds = predictions[0], predictions[1]

    class_accuracy = accuracy_score(y_test[:, 0], np.argmax(class_preds, axis=1))
    level_mae = mean_absolute_error(y_test[:, 1], level_preds)

    return class_accuracy, level_mae, np.argmax(class_preds, axis=1), level_preds

def load_test_data(test_dir):
    all_fMRI_data, all_ppg_data, all_resp_data, participant_trial_pairs = [], [], [], []
    for participant in os.listdir(test_dir):
        participant_path = os.path.join(test_dir, participant)
        if os.path.isdir(participant_path):
            fMRI_data = np.load(os.path.join(participant_path, 'fMRI_data.npz'))['data']
            ppg_data = np.load(os.path.join(participant_path, 'ppg_data.npz'))['data']
            resp_data = np.load(os.path.join(participant_path, 'resp_data.npz'))['data']
            fMRI_data = handle_missing_values(fMRI_data)
            ppg_data = handle_missing_values(ppg_data)
            resp_data = handle_missing_values(resp_data)
            all_fMRI_data.append(fMRI_data)
            all_ppg_data.append(ppg_data)
            all_resp_data.append(resp_data)
            for trial in range(fMRI_data.shape[0]):
                participant_trial_pairs.append((participant, trial + 1))

    all_fMRI_data = np.concatenate(all_fMRI_data, axis=0)
    all_ppg_data = np.concatenate(all_ppg_data, axis=0)
    all_resp_data = np.concatenate(all_resp_data, axis=0)

    return all_fMRI_data, all_ppg_data, all_resp_data, participant_trial_pairs

def rescale_level_preds(level_preds, min_level, max_level):
    # Apply the scaling back to the original level range
    return (level_preds * (max_level - min_level)) + min_level

if __name__ == "__main__":
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Test'))
    submission_template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Supplementary', 'submission.csv'))
    output_submission_path = 'submission.csv'

    # Load model
    model = build_model()
    model.load_weights('best_model.keras')

    # Load test data
    fMRI_data, ppg_data, resp_data, participant_trial_pairs = load_test_data(test_dir)
    fMRI_data = fMRI_data[..., np.newaxis]  # Shape (samples, 246, 25, 1)
    ppg_data = ppg_data[..., np.newaxis]  # Shape (samples, 10000, 1)
    resp_data = resp_data[..., np.newaxis]  # Shape (samples, 10000, 1)

    # Make predictions
    class_preds, level_preds = model.predict([fMRI_data, ppg_data, resp_data])
    class_preds = np.argmax(class_preds, axis=1)

    # Rescale level predictions
    min_level, max_level = -4, 4  # These should be set based on training
    level_preds = rescale_level_preds(level_preds, min_level, max_level)

    # Load the submission template
    submission_df = pd.read_csv(submission_template_path)

    # Fill the submission template with predictions
    for i, (participant, trial) in enumerate(participant_trial_pairs):
        submission_df.loc[(submission_df['Participant'] == participant) & (submission_df['Trial'] == trial), 'CLASS'] = class_preds[i]
        submission_df.loc[(submission_df['Participant'] == participant) & (submission_df['Trial'] == trial), 'LEVEL'] = level_preds[i]

    # Save the filled submission
    submission_df.to_csv(output_submission_path, index=False)
    print(f"Predictions saved to {output_submission_path}")
