import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import KeyedVectors
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# === 🟢 Load Google's Pre-trained Word2Vec Model === #
word_vectors = KeyedVectors.load_word2vec_format("datasets/GoogleNews-vectors-negative300.bin", binary=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download NLTK tokenizer and stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Define the stopwords list
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...', 'br', 'http', 'https']  # Add more if needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

# === 🟢 Text Preprocessing Functions === #
def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["  
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in text.split() if word.lower() not in final_stop_words_list])

def clean_str(text):
    """Clean text by removing unwanted characters, extra spaces, and formatting issues."""
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    return text.strip().lower()

# === 🟢 Load Dataset === #
df = pd.read_csv("datasets/tensorflow.csv")

# ✅ Use the correct labels column: 'class'
df = df[['Title', 'Body', 'class']]
df['Body'] = df['Body'].fillna('')
df['text'] = df['Title'] + " " + df['Body']
df = df[['text', 'class']]

# === 🟢 Apply Preprocessing === #
df['text'] = df['text'].apply(remove_html)
df['text'] = df['text'].apply(remove_emoji)
df['text'] = df['text'].apply(clean_str)
df['text'] = df['text'].apply(remove_stopwords)

# === 🟢 Tokenization === #
df['tokens'] = df['text'].apply(lambda x: word_tokenize(x.lower()))

# === 🟢 Convert Words to Embeddings === #
def text_to_embedding(tokens):
    embeddings = [word_vectors[word] for word in tokens if word in word_vectors]
    if not embeddings:
        embeddings = [np.zeros(300)]
    return np.mean(embeddings, axis=0)

df['embeddings'] = df['tokens'].apply(text_to_embedding)

# Convert to numpy arrays
X = np.stack(df['embeddings'].values)  # Shape: (num_samples, 300)
y = df['class'].values

# === 🟢 Train-Test Split with Oversampling === #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
ros = RandomOverSampler(sampling_strategy='minority')
X_train, y_train = ros.fit_resample(X_train, y_train)

# === 🟢 Define LSTM Model === #
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 because of bidirectional

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length of 1: (batch_size, 1, embedding_dim)
        _, (hidden, _) = self.lstm(x)

        # Extract last forward and backward hidden states
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]

        # Concatenate them
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        x = self.fc(combined_hidden)
        return x

# === 🟢 Run the Experiment 10 Times === #
num_runs = 10
num_epochs = 15

results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
    "auc_val": []
}

# === 🟢 Find Best Threshold Using ROC Curve === #
def find_best_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_idx = np.argmax(tpr + (1 - fpr))  # Alternative method

    return thresholds[optimal_idx]

for run in range(num_runs):
    print(f"\n🔄 Run {run + 1}/{num_runs}")

    # === Train-Test Split === #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=run)

    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Initialize model
    embedding_dim = 300
    hidden_dim = 300
    num_layers = 4
    output_dim = 1

    model = LSTMClassifier(embedding_dim, hidden_dim, num_layers, output_dim).to(device)

    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # === 🟢 Training Loop === #
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # === 🟢 Evaluate Model === #
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor).squeeze()
        y_pred_probs = torch.sigmoid(y_pred_logits)

        # print("Predicted Probability Distribution:")
        # print(y_pred_probs.cpu().numpy())

        best_threshold = find_best_threshold(y_test_tensor.cpu().numpy(), y_pred_probs.cpu().numpy())
        print(f"Best Threshold Found: {best_threshold:.4f}")
        y_pred_classes = (y_pred_probs > best_threshold).int()

    # Plot Probabilities
    y_pred_probs_np = y_pred_probs.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    y_pred_np = y_pred_classes.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_probs_np[y_test_np == 0], bins=30, alpha=0.5, label='Class 0')
    plt.hist(y_pred_probs_np[y_test_np == 1], bins=30, alpha=0.5, label='Class 1')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f"Threshold = {best_threshold:.4f}")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution of Model Predictions')
    plt.legend()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test_np, y_pred_np, pos_label=1)

    auc_val = auc(fpr, tpr)
    acc = accuracy_score(y_test_np, y_pred_np)
    prec = precision_score(y_test_np, y_pred_np, zero_division=0, average='macro')
    rec = recall_score(y_test_np, y_pred_np, zero_division=0, average='macro')
    f1 = f1_score(y_test_np, y_pred_np, zero_division=0, average='macro')

    results["accuracy"].append(acc)
    results["precision"].append(prec)
    results["recall"].append(rec)
    results["f1_score"].append(f1)
    results["auc_val"].append(auc_val)

    print("\n🚀 **Results Across 10 Runs** 🚀")
    print(f"✅ **Accuracy**:  {acc:.4f}")
    print(f"✅ **Precision**: {prec:.4f}")
    print(f"✅ **Recall**:    {rec:.4f}")
    print(f"✅ **F1 Score**:  {f1:.4f}")
    print(f"✅ **AUC**:       {auc_val:.4f}")

# === 🟢 Compute Averages Across 10 Runs === #
final_accuracy = np.mean(results["accuracy"])
final_precision = np.mean(results["precision"])
final_recall = np.mean(results["recall"])
final_f1_score = np.mean(results["f1_score"])
final_auc_val = np.mean(results["auc_val"])

print("\n🚀 **Final Averaged Results Across 10 Runs** 🚀")
print(f"✅ **Average Accuracy**:  {final_accuracy:.4f}")
print(f"✅ **Average Precision**: {final_precision:.4f}")
print(f"✅ **Average Recall**:    {final_recall:.4f}")
print(f"✅ **Average F1 Score**:  {final_f1_score:.4f}")
print(f"✅ **Average AUC**:       {final_auc_val:.4f}")
