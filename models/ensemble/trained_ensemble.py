import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path

dev_d = np.load("models/distilbert/results/dev_probs.npy")
test_d = np.load("models/distilbert/results/test_probs.npy")
dev_lr = np.load("models/logistic_regression/results/dev_probs.npy")
test_lr = np.load("models/logistic_regression/results/test_probs.npy")
dev_y = np.load("models/logistic_regression/results/dev_labels.npy")
test_y = np.load("models/logistic_regression/results/test_labels.npy")

best_w, best_f1 = 0, 0
for w in np.arange(0, 1.05, 0.05):
    blended = w * dev_d + (1 - w) * dev_lr
    f1 = f1_score(dev_y, blended >= 0.5)
    if f1 > best_f1:
        best_w, best_f1 = w, f1

final = best_w * test_d + (1 - best_w) * test_lr
preds = final >= 0.5

print(f"Best weight (DistilBERT): {best_w:.2f}")
print(classification_report(test_y, preds, target_names=["not sexist", "sexist"]))
print(confusion_matrix(test_y, preds))

results_dir = Path("models/ensemble/results")
results_dir.mkdir(parents=True, exist_ok=True)
np.save(results_dir / "ensemble_test_probs.npy", final)
with open(results_dir / "classification_report.txt", "w") as f:
    f.write(
        classification_report(test_y, preds, target_names=["not sexist", "sexist"])
        + "\n"
    )
