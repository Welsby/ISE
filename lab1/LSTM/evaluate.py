import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from config import device


def find_best_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        y_pred_logits = model(X_test_tensor).squeeze()
        y_pred_probs = torch.sigmoid(y_pred_logits)
        best_threshold = find_best_threshold(y_test, y_pred_probs.cpu().numpy())
        y_pred_classes = (y_pred_probs > best_threshold).int()

    acc = accuracy_score(y_test, y_pred_classes.cpu().numpy())
    prec = precision_score(y_test, y_pred_classes.cpu().numpy(), average='macro')
    rec = recall_score(y_test, y_pred_classes.cpu().numpy(), average='macro')
    f1 = f1_score(y_test, y_pred_classes.cpu().numpy(), average='macro')

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    return acc, prec, rec, f1
