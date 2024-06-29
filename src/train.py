import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import build_model, get_callbacks
from preprocessing import normalize_data, handle_missing_values, preprocess_labels, scale_level_labels, augment_fMRI_data, augment_ppg_data, augment_resp_data

def train_model(data_dir, epochs=20, batch_size=32, learning_rate=1e-4):
    try:
        # Load data from all subjects in the training directory
        all_fMRI_data, all_ppg_data, all_resp_data, all_labels = [], [], [], []

        for subject_folder in os.listdir(data_dir):
            subject_path = os.path.join(data_dir, subject_folder)
            if os.path.isdir(subject_path):
                fMRI_data, ppg_data, resp_data, labels = load_data(subject_path)
                if fMRI_data is not None and ppg_data is not None and resp_data is not None and labels is not None:
                    all_fMRI_data.append(fMRI_data)
                    all_ppg_data.append(ppg_data)
                    all_resp_data.append(resp_data)
                    all_labels.append(labels)

        if not all_fMRI_data or not all_ppg_data or not all_resp_data or not all_labels:
            print(f"Data loading failed for {data_dir}.")
            return None, None, None

        # Combine data from all subjects
        fMRI_data = np.concatenate(all_fMRI_data, axis=0)
        ppg_data = np.concatenate(all_ppg_data, axis=0)
        resp_data = np.concatenate(all_resp_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Preprocess data
        fMRI_data, ppg_data, resp_data = normalize_data(fMRI_data, ppg_data, resp_data)
        fMRI_data = handle_missing_values(fMRI_data)
        ppg_data = handle_missing_values(ppg_data)
        resp_data = handle_missing_values(resp_data)

        labels, valid_indices = preprocess_labels(labels)
        fMRI_data = fMRI_data[valid_indices]
        ppg_data = ppg_data[valid_indices]
        resp_data = resp_data[valid_indices]

        class_labels, level_labels = labels[:, 0], labels[:, 1]
        labels, min_level, max_level = scale_level_labels(labels)

        min_length = min(len(fMRI_data), len(ppg_data), len(resp_data), len(class_labels), len(level_labels))
        fMRI_data = fMRI_data[:min_length]
        ppg_data = ppg_data[:min_length]
        resp_data = resp_data[:min_length]
        class_labels = class_labels[:min_length]
        level_labels = level_labels[:min_length]

        fMRI_aug_data = augment_fMRI_data(fMRI_data)
        ppg_aug_data = augment_ppg_data(ppg_data)
        resp_aug_data = augment_resp_data(resp_data)

        fMRI_data = np.concatenate((fMRI_data, fMRI_aug_data), axis=0)
        ppg_data = np.concatenate((ppg_data, ppg_aug_data), axis=0)
        resp_data = np.concatenate((resp_data, resp_aug_data), axis=0)
        class_labels = np.concatenate((class_labels, class_labels), axis=0)
        level_labels = np.concatenate((level_labels, level_labels), axis=0)

        fMRI_train, fMRI_val, ppg_train, ppg_val, resp_train, resp_val, class_train, class_val, level_train, level_val = train_test_split(
            fMRI_data, ppg_data, resp_data, class_labels, level_labels, test_size=0.2, random_state=42)

        if not (fMRI_val.size and ppg_val.size and resp_val.size):
            print("Validation data is empty. Please check the data loader and preprocessing steps.")
            return None, None, None

        model = build_model()
        callbacks = get_callbacks()

        model.fit(
            [fMRI_train, ppg_train, resp_train],
            [class_train, level_train],
            validation_data=([fMRI_val, ppg_val, resp_val], [class_val, level_val]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return model, min_level, max_level
    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
        return None, None, None

if __name__ == "__main__":
    data_dir = "data/Train"
    model, min_level, max_level = train_model(data_dir)
