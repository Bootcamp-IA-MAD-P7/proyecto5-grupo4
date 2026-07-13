import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_ensemble_data():
    dev_d = np.load("models/distilbert/results/dev_probs.npy")
    test_d = np.load("models/distilbert/results/test_probs.npy")
    dev_lr = np.load("models/logistic_regression/results/dev_probs.npy")
    test_lr = np.load("models/logistic_regression/results/test_probs.npy")
    dev_y = np.load("models/logistic_regression/results/dev_labels.npy")
    test_y = np.load("models/logistic_regression/results/test_labels.npy")

    return dev_d, test_d, dev_lr, test_lr, dev_y, test_y


def find_best_weight(dev_d, dev_lr, dev_y):
    best_w, best_f1 = 0, 0
    for w in np.arange(0, 1.05, 0.05):
        blended = w * dev_d + (1 - w) * dev_lr
        f1 = f1_score(dev_y, blended >= 0.5)
        if f1 > best_f1:
            best_w, best_f1 = w, f1

    return best_w, best_f1


def evaluate_ensemble(best_w, test_d, test_lr, test_y):
    final = best_w * test_d + (1 - best_w) * test_lr
    preds = final >= 0.5

    print(f"Best weight (DistilBERT): {best_w:.2f}")
    print(classification_report(test_y, preds, target_names=["not sexist", "sexist"]))
    print(confusion_matrix(test_y, preds))

    return final, preds


def save_ensemble_results(best_w, final, test_y, preds):
    results_dir = Path("models/ensemble/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "ensemble_test_probs.npy", final)

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(
            classification_report(test_y, preds, target_names=["not sexist", "sexist"])
            + "\n"
        )

    weights = {
        "distilbert_weight": float(best_w),
        "logistic_regression_weight": float(1 - best_w),
    }
    with open("models/ensemble/weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nEnsemble weights saved to models/ensemble/weights.json")


def main():
    print("Loading ensemble data...")
    dev_d, test_d, dev_lr, test_lr, dev_y, test_y = load_ensemble_data()

    print("Finding best weight...")
    best_w, best_f1 = find_best_weight(dev_d, dev_lr, dev_y)
    print(f"Best dev F1: {best_f1:.4f}")

    print("\nEvaluating on test set...")
    final, preds = evaluate_ensemble(best_w, test_d, test_lr, test_y)

    print("\nSaving results...")
    save_ensemble_results(best_w, final, test_y, preds)


if __name__ == "__main__":
    main()
