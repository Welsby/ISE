import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from data_preprocessing import load_and_preprocess_data
from model import LSTMClassifier
from evaluate import evaluate_model
from config import device, num_epochs


def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 0.01)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=8)

    # Load and preprocess data
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    ros = RandomOverSampler(sampling_strategy='minority')
    X_train, y_train = ros.fit_resample(X_train, y_train)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(embedding_dim=300, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

    acc, _, _, _ = evaluate_model(model, X_test, y_test)
    return acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Hyperparameters:", study.best_params)
    print("Best Accuracy:", study.best_value)
