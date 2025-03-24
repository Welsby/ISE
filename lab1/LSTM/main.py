from data_preprocessing import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model
from config import num_runs

if __name__ == "__main__":
    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    for run in range(num_runs):
        print(f"\n🔄 Run {run + 1}/{num_runs}")
        X, y = load_and_preprocess_data()
        model, X_test, y_test = train_model(X, y)
        acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)

        results["accuracy"].append(acc)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1_score"].append(f1)

    print("\n🚀 **Final Averaged Results Across Runs** 🚀")
    print(f"✅ **Average Accuracy**:  {sum(results['accuracy']) / num_runs:.4f}")
    print(f"✅ **Average Precision**: {sum(results['precision']) / num_runs:.4f}")
    print(f"✅ **Average Recall**:    {sum(results['recall']) / num_runs:.4f}")
    print(f"✅ **Average F1 Score**:  {sum(results['f1_score']) / num_runs:.4f}")