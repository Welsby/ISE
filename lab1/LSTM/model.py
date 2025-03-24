import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        x = self.fc(combined_hidden)
        return x
