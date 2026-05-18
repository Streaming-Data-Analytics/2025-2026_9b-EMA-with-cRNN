import torch
import torch.nn as nn

class cLSTM(nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        # Fully connected layer that acts as classifier head (maps last hidden state to class scores)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    # cLSTM model call
    def forward(self, x):
        out, _ = self.lstm(x)
        # many to one: takes all batches, last timestep and all hidden features
        # then map to 2 class scores per sample in the batch
        return self.fc(out[:, -1, :])   # many-to-one: only last element

class cGRU(nn.Module):
    #Constructor
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
        # Fully connected layer that acts as classifier head (maps last hidden state to class scores)
        self.fc  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
