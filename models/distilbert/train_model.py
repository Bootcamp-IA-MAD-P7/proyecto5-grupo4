import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from pathlib import Path

_base = Path(__file__).resolve().parent.parent


class SexismDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.as_tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        ce_loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def load_data():
    train_data = load_file(_base / "models" / "train_data.safetensors")
    dev_data = load_file(_base / "models" / "dev_data.safetensors")
    test_data = load_file(_base / "models" / "test_data.safetensors")
    config = torch.load(_base / "models" / "config.pt", weights_only=False)

    train_dataset = SexismDataset(
        encodings={
            "input_ids": train_data["input_ids"],
            "attention_mask": train_data["attention_mask"],
        },
        labels=train_data["labels"].tolist(),
    )
    dev_dataset = SexismDataset(
        encodings={
            "input_ids": dev_data["input_ids"],
            "attention_mask": dev_data["attention_mask"],
        },
        labels=dev_data["labels"].tolist(),
    )
    test_dataset = SexismDataset(
        encodings={
            "input_ids": test_data["input_ids"],
            "attention_mask": test_data["attention_mask"],
        },
        labels=test_data["labels"].tolist(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    return train_loader, dev_loader, test_loader, config


def load_model(num_labels=2):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    f1 = f1_score(labels, predictions, average="binary")
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    return {"f1": f1, "precision": precision, "recall": recall}


class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = FocalLoss(alpha=self.class_weights, gamma=self.gamma)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    GAMMA = 1.0

    print("Loading data...")
    train_loader, dev_loader, test_loader, config = load_data()

    print("Loading model...")
    model = load_model(num_labels=2)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    trainer = CustomTrainer(
        class_weights=config["class_weights"],
        gamma=GAMMA,
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=dev_loader.dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    print(f"Training with Focal Loss (gamma={GAMMA})...")
    trainer.train()

    print("\nTuning threshold on dev set...")
    dev_logits = trainer.predict(dev_loader.dataset).predictions
    dev_probs = torch.nn.functional.softmax(torch.tensor(dev_logits), dim=-1)[
        :, 1
    ].numpy()

    precisions, recalls, thresholds = precision_recall_curve(
        dev_loader.dataset.labels, dev_probs
    )
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    best_dev_f1 = f1_scores[best_idx]
    print(f"Best threshold: {best_threshold:.3f} (dev F1: {best_dev_f1:.4f})")

    print("\nEvaluating on test set...")
    test_logits = trainer.predict(test_loader.dataset).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1)[
        :, 1
    ].numpy()
    preds = (test_probs >= best_threshold).astype(int)
    true_labels = test_loader.dataset.labels

    print("Threshold=0.5 results (default):")
    preds_default = test_logits.argmax(axis=-1)
    print(
        classification_report(
            true_labels, preds_default, target_names=["not sexist", "sexist"]
        )
    )
    cm_default = confusion_matrix(true_labels, preds_default)
    print(f"Confusion Matrix:\n{cm_default}")

    print(f"\nThreshold={best_threshold:.3f} results (tuned):")
    print(
        classification_report(true_labels, preds, target_names=["not sexist", "sexist"])
    )
    cm = confusion_matrix(true_labels, preds)
    print(f"Confusion Matrix:\n{cm}")

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "confusion_matrix.txt", "w") as f:
        f.write("Threshold=0.5 (default):\n")
        f.write(str(cm_default) + "\n\n")
        f.write(f"Threshold={best_threshold:.3f} (tuned):\n")
        f.write(str(cm) + "\n")

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write("Threshold=0.5 (default):\n")
        f.write(
            classification_report(
                true_labels, preds_default, target_names=["not sexist", "sexist"]
            )
            + "\n\n"
        )
        f.write(f"Threshold={best_threshold:.3f} (tuned):\n")
        f.write(
            classification_report(
                true_labels, preds, target_names=["not sexist", "sexist"]
            )
            + "\n"
        )

    metrics = {
        "gamma": GAMMA,
        "best_threshold": float(best_threshold),
        "dev_best_f1": float(best_dev_f1),
        "default_f1": float(f1_score(true_labels, preds_default)),
        "tuned_f1": float(f1_score(true_labels, preds)),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(results_dir / "test_probs.npy", test_probs)
    np.save(results_dir / "dev_probs.npy", dev_probs)

    print(f"\nResults saved to {results_dir}/")
