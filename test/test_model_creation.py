# test/test_model_creation.py
import os
import sys

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import build_model

# Build the model
model = build_model()

# Print the model summary to verify its architecture
model.summary()
