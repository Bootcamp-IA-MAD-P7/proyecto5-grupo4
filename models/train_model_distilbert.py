import os
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pathlib import Path

_base = Path(__file__).resolve().parent.parent


class SexismDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


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


if __name__ == "__main__":
    print("Loading data...")
    train_loader, dev_loader, test_loader, config = load_data()

    print("Loading model...")
    model = load_model(num_labels=2)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
