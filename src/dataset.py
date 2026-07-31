import torch
from sklearn.datasets import make_swiss_roll

def get_swiss_roll_data(batch_size=1024):
    """Generates a batch of 2D Swiss Roll points."""
    x, _ = make_swiss_roll(batch_size, noise=0.5)
    data = x[:, [0, 2]]
    # Normalize data to roughly [-1, 1] bounds
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)