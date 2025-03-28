from scipy.stats import ttest_rel

# Accuracy
your_model_acc = [0.8479, 0.7469, 0.7672, 0.8006, 0.6326]
baseline_acc =    [0.5631, 0.6238, 0.5567, 0.6077, 0.5224]

# Precision
your_model_pre = [0.768, 0.6421, 0.6985, 0.6737, 0.6002]
baseline_pre = [0.6358, 0.6056, 0.6285, 0.6138, 0.5571]

# Recall
your_model_recall = [0.8739, 0.7798, 0.7721, 0.811, 0.7400]
baseline_recall = [0.7226, 0.7402, 0.6961, 0.7505, 0.6234]

# F1 Score
your_model_f1 = [0.7963, 0.6457, 0.7086, 0.6961, 0.5502]
baseline_f1 = [0.5406, 0.5519, 0.5369, 0.5479, 0.4428]

# Run t-tests
metrics = {
    "Accuracy": (your_model_acc, baseline_acc),
    "Precision": (your_model_pre, baseline_pre),
    "Recall": (your_model_recall, baseline_recall),
    "F1 Score": (your_model_f1, baseline_f1)
}

# Print results with significance
for name, (model, baseline) in metrics.items():
    t_stat, p_value = ttest_rel(model, baseline)
    significance = "✅ Statistically significant" if p_value < 0.05 else "❌ Not statistically significant"
    print(f"{name} - T-statistic: {t_stat:.4f}, P-value: {p_value:.4f} → {significance}")
