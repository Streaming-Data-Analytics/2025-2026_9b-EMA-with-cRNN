import torch
import numpy as np
from torch.utils.data import Dataset

def create_windowed_dataset(df, feature_cols, target_col, task_col, window_size, hop_size):

    X, y, tasks = [], [], []
    values  = df[feature_cols].values # shape (N, num_features)
    targets = df[target_col].values # shape (N,)
    task_ids = df[task_col].values # shape (N,)

    for i in range(0, len(values) - window_size + 1, hop_size):
        X.append(values[i : i + window_size]) # (W, num_features)
        y.append(targets[i + window_size - 1]) # label for last step
        tasks.append(task_ids[i + window_size - 1]) # task at prediction point

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(tasks)

# Create a dataset object from the windowed dataset
class StreamingWindowDataset(Dataset):
    def __init__(self, X, y, tasks):
        self.X = torch.tensor(X) # (N, W, num_features)
        self.y = torch.tensor(y) # (N,)  int64 for CrossEntropy
        self.tasks = tasks # numpy array, for drift detection

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.tasks[idx]
