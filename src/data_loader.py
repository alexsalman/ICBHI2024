# src/data_loader.py
import os
import numpy as np

def load_data(data_dir):
    try:
        fMRI_data = []
        ppg_data = []
        resp_data = []
        labels = []

        for participant in os.listdir(data_dir):
            participant_path = os.path.join(data_dir, participant)
            if os.path.isdir(participant_path):
                fMRI_data.append(np.load(os.path.join(participant_path, 'fMRI_data.npz'))['data'])
                ppg_data.append(np.load(os.path.join(participant_path, 'ppg_data.npz'))['data'])
                resp_data.append(np.load(os.path.join(participant_path, 'resp_data.npz'))['data'])
                labels.append(np.load(os.path.join(participant_path, 'labels.npz'))['data'])

        if not (fMRI_data and ppg_data and resp_data and labels):
            print("Data is missing. Ensure all npz files are present and correctly named.")
            return None, None, None, None

        fMRI_data = np.concatenate(fMRI_data, axis=0)
        ppg_data = np.concatenate(ppg_data, axis=0)
        resp_data = np.concatenate(resp_data, axis=0)
        labels = np.concatenate(labels, axis=0)

        return fMRI_data, ppg_data, resp_data, labels

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Check if the file exists at {data_dir}.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None
