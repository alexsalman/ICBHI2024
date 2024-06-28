# src/train.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import build_model, get_callbacks
from preprocessing import normalize_data, handle_missing_values, preprocess_labels, scale_level_labels, augment_fMRI_data, augment_ppg_data, augment_resp_data

def train_model(data_dir, epochs=20, batch_size=32, learning_rate=1e-4):
    try:
        print(f"Loading data from {data_dir}...")
        fMRI_data, ppg_data, resp_data, labels = load_data(data_dir)

        if fMRI_data is None or ppg_data is None or resp_data is None or labels is None:
            print(f"Data loading failed for {data_dir}.")
            return None, None, None

        print("Normalizing and handling missing values...")
        fMRI_data, ppg_data, resp_data = normalize_data(fMRI_data, ppg_data, resp_data)
        fMRI_data = handle_missing_values(fMRI_data)
        ppg_data = handle_missing_values(ppg_data)
        resp_data = handle_missing_values(resp_data)

        print("Preprocessing labels...")
        labels, valid_indices = preprocess_labels(labels)
        fMRI_data = fMRI_data[valid_indices]
        ppg_data = ppg_data[valid_indices]
        resp_data = resp_data[valid_indices]

        class_labels, level_labels = labels[:, 0], labels[:, 1]
        labels, min_level, max_level = scale_level_labels(labels)

        print("Ensuring data length consistency...")
        min_length = min(len(fMRI_data), len(ppg_data), len(resp_data), len(class_labels), len(level_labels))
        fMRI_data = fMRI_data[:min_length]
        ppg_data = ppg_data[:min_length]
        resp_data = resp_data[:min_length]
        class_labels = class_labels[:min_length]
        level_labels = level_labels[:min_length]

        print("Augmenting data...")
        fMRI_aug_data = augment_fMRI_data(fMRI_data)
        ppg_aug_data = augment_ppg_data(ppg_data)
        resp_aug_data = augment_resp_data(resp_data)

        fMRI_data = np.concatenate((fMRI_data, fMRI_aug_data), axis=0)
        ppg_data = np.concatenate((ppg_data, ppg_aug_data), axis=0)
        resp_data = np.concatenate((resp_data, resp_aug_data), axis=0)
        class_labels = np.concatenate((class_labels, class_labels), axis=0)
        level_labels = np.concatenate((level_labels, level_labels), axis=0)

        print("Splitting data into training and validation sets...")
        fMRI_train, fMRI_val, ppg_train, ppg_val, resp_train, resp_val, class_train, class_val, level_train, level_val = train_test_split(
            fMRI_data, ppg_data, resp_data, class_labels, level_labels, test_size=0.2, random_state=42)

        if not (fMRI_val.size and ppg_val.size and resp_val.size):
            print("Validation data is empty. Please check the data loader and preprocessing steps.")
            return None, None, None

        print("Building model...")
        model = build_model()

        print("Training model...")
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
    data_dir = "/projects/2024 ICBHI Scientific Challenge/ICBHI2024/data/Train"  # Update this with the correct path
    model, min_level, max_level = train_model(data_dir)
