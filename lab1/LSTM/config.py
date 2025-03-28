import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
embedding_dim = 300
hidden_dim = 192
num_layers = 2
output_dim = 1
batch_size = 32
num_epochs = 5
learning_rate = 0.0007
num_runs = 10
