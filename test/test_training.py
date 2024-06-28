# test/test_training.py
import os
import sys

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import train_model

# Set the path to the directory containing the data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'Train')

# Train the model
model = train_model(data_dir, epochs=1, batch_size=2)
