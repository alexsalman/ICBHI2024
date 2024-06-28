# src/main.py
import os
import numpy as np
import pandas as pd
from train import train_model
from evaluate import evaluate_model, load_test_data, rescale_level_preds
from data_loader import load_data
from predict import predict

def main():
    print(f"Current working directory: {os.getcwd()}")

    data_dir = '../ICBHI2024/data/Train'
    test_dir = '../ICBHI2024/data/Test'
    submission_template_path = '../ICBHI2024/data/Supplementary/submission.csv'
    output_submission_path = '../ICBHI2024/submission.csv'

    # Verify directories exist
    if not os.path.exists(data_dir):
        print(f"Training data directory does not exist: {data_dir}")
        return
    if not os.path.exists(test_dir):
        print(f"Test data directory does not exist: {test_dir}")
        return

    # Train the model
    print("Starting training...")
    model, min_level, max_level = train_model(data_dir, epochs=10, batch_size=8)

    if model is None:
        print("Training failed.")
        return

    # Load validation data
    val_subject_path = os.path.join(data_dir, 'P16')
    fMRI_data, ppg_data, resp_data, labels = load_data(val_subject_path)

    if fMRI_data is None or ppg_data is None or resp_data is None or labels is None:
        print("Validation data loading failed.")
        return

    class_labels, level_labels = labels[:, 0], labels[:, 1]

    fMRI_data = fMRI_data[..., np.newaxis]  # Shape (samples, 246, 25, 1)
    ppg_data = ppg_data[..., np.newaxis]  # Shape (samples, 10000, 1)
    resp_data = resp_data[..., np.newaxis]  # Shape (samples, 10000, 1)

    # Evaluate the model on validation set
    print("Evaluating the model on validation set...")
    try:
        class_accuracy, level_mae, class_preds, level_preds = evaluate_model(model, [fMRI_data, ppg_data, resp_data], labels)
        print(f'Validation Class Accuracy: {class_accuracy}, Validation Level MAE: {level_mae}')
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

    # Predict on the test set
    print("Predicting on the test set...")
    test_subjects = ['P17', 'P18', 'P19', 'P20']
    try:
        submission_df = pd.read_csv(submission_template_path)

        for subject in test_subjects:
            subject_path = os.path.join(test_dir, subject)
            class_preds, level_preds = predict(model, subject_path, min_level, max_level)

            if class_preds is not None and level_preds is not None:
                for i in range(len(class_preds)):
                    submission_df.loc[(submission_df['Participant'] == subject) & (submission_df['Trial'] == (i + 1)), 'CLASS'] = class_preds[i]
                    submission_df.loc[(submission_df['Participant'] == subject) & (submission_df['Trial'] == (i + 1)), 'LEVEL'] = level_preds[i]

        # Save the filled submission
        submission_df.to_csv(output_submission_path, index=False)
        print(f"Predictions saved to {output_submission_path}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
