# src/data_loader.py
import os
import numpy as np  # Import numpy

def load_data(subject_path):
    try:
        fMRI_data = np.load(os.path.join(subject_path, 'fMRI_data.npz'))['data']
        ppg_data = np.load(os.path.join(subject_path, 'PPG_data.npz'))['data']
        resp_data = np.load(os.path.join(subject_path, 'Resp_data.npz'))['data']
        labels_path = os.path.join(subject_path, 'labels.npz')
        if os.path.exists(labels_path):
            labels = np.load(labels_path)['data']
        else:
            labels = None
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Check if the file exists.")
        return None, None, None, None
    except KeyError as e:
        print(f"KeyError: {e}. Check if 'data' key exists in the npz file.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

    return fMRI_data, ppg_data, resp_data, labels
