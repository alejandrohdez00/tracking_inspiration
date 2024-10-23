import os
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
