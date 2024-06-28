# src/preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_data(fMRI_data, ppg_data, resp_data):
    fMRI_scaler = StandardScaler()
    ppg_scaler = StandardScaler()
    resp_scaler = StandardScaler()

    fMRI_data = fMRI_data.reshape(-1, fMRI_data.shape[-2])
    fMRI_data = fMRI_scaler.fit_transform(fMRI_data)
    fMRI_data = fMRI_data.reshape(-1, 246, 25, 1)

    ppg_data = ppg_data.reshape(-1, ppg_data.shape[-2])
    ppg_data = ppg_scaler.fit_transform(ppg_data)
    ppg_data = ppg_data.reshape(-1, 10000, 1)

    resp_data = resp_data.reshape(-1, resp_data.shape[-2])
    resp_data = resp_scaler.fit_transform(resp_data)
    resp_data = resp_data.reshape(-1, 10000, 1)

    return fMRI_data, ppg_data, resp_data

def handle_missing_values(data):
    mask = np.isnan(data)
    data[mask] = np.nanmean(data)
    return data

def preprocess_labels(labels):
    valid_indices = np.all(labels != -1, axis=1)
    labels = labels[valid_indices]
    return labels, valid_indices

def scale_level_labels(labels):
    level_labels = labels[:, 1]
    min_level, max_level = level_labels.min(), level_labels.max()
    level_labels = (level_labels - min_level) / (max_level - min_level)
    labels[:, 1] = level_labels
    return labels, min_level, max_level

def rescale_level_preds(level_preds, min_level, max_level):
    level_preds = level_preds * (max_level - min_level) + min_level
    return level_preds

def augment_fMRI_data(fMRI_data):
    noise_factor = 0.1
    fMRI_data_noisy = fMRI_data + noise_factor * np.random.randn(*fMRI_data.shape)
    return fMRI_data_noisy

def augment_ppg_data(ppg_data):
    noise_factor = 0.1
    ppg_data_noisy = ppg_data + noise_factor * np.random.randn(*ppg_data.shape)
    return ppg_data_noisy

def augment_resp_data(resp_data):
    noise_factor = 0.1
    resp_data_noisy = resp_data + noise_factor * np.random.randn(*resp_data.shape)
    return resp_data_noisy

def augment_flip_fMRI(fMRI_data):
    return np.flip(fMRI_data, axis=2)

def augment_flip_ppg(ppg_data):
    return np.flip(ppg_data, axis=1)

def augment_flip_resp(resp_data):
    return np.flip(resp_data, axis=1)

def augment_shift_fMRI(fMRI_data, shift=2):
    return np.roll(fMRI_data, shift, axis=1)

def augment_shift_ppg(ppg_data, shift=100):
    return np.roll(ppg_data, shift, axis=1)

def augment_shift_resp(resp_data, shift=100):
    return np.roll(resp_data, shift, axis=1)

def augment_flip_vertical_fMRI(fMRI_data):
    return np.flip(fMRI_data, axis=1)

def augment_flip_vertical_ppg(ppg_data):
    return np.flip(ppg_data, axis=0)

def augment_flip_vertical_resp(resp_data):
    return np.flip(resp_data, axis=0)

def augment_noise_fMRI(fMRI_data):
    return augment_fMRI_data(fMRI_data)

def augment_noise_ppg(ppg_data):
    return augment_ppg_data(ppg_data)

def augment_noise_resp(resp_data):
    return augment_resp_data(resp_data)
