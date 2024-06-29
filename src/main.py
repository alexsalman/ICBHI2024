import os
import numpy as np
import pandas as pd
from train import train_model
from evaluate import evaluate_model, load_and_prepare_data
from data_loader import load_data
from predict import predict

def main():
    print(f"Current working directory: {os.getcwd()}")

    data_dir = 'data/Train'
    test_dir = 'data/Test'
    submission_template_path = 'data/Supplementary/submission.csv'
    output_submission_path = 'submission.csv'

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
    val_data, val_labels = load_and_prepare_data(val_subject_path)

    if val_data is None or val_labels is None:
        print("Validation data loading failed.")
        return

    # Evaluate the model on validation set
    print("Evaluating the model on validation set...")
    try:
        class_accuracy, level_rmse, class_preds, level_preds = evaluate_model(model, val_data, val_labels, min_level, max_level)
        print(f'Validation Class Accuracy: {class_accuracy}, Validation Level RMSE: {level_rmse}')
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
