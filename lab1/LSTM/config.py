import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
embedding_dim = 300
hidden_dim = 300
num_layers = 4
output_dim = 1
batch_size = 128
num_epochs = 15
learning_rate = 0.005
num_runs = 10
