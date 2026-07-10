import os
from pathlib import Path
import re
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.utils.class_weight import compute_class_weight
from safetensors.torch import save_file, load_file


try:
    _base = Path(__file__).resolve().parent.parent
except NameError:
    _base = Path(os.getcwd()).parent
df = pl.read_csv(_base / "data" / "raw" / "edos_labelled_aggregated.csv")
df_pd = df.to_pandas().reset_index()

print(df["split"].value_counts())

train_df = df.filter(pl.col("split") == "train")
dev_df = df.filter(pl.col("split") == "dev")
test_df = df.filter(pl.col("split") == "test")

print(f"Train: {train_df.shape}")
print(f"Dev:   {dev_df.shape}")
print(f"Test:  {test_df.shape}")


train_df = train_df.with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
dev_df = dev_df.with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
test_df = test_df.with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)

print(train_df["label"].value_counts())
print(dev_df["label"].value_counts())
print(test_df["label"].value_counts())


# Normalize in lowercase, remove meaningless placeholders [USER], [URL], fix extra spaces
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[USER\]", "", text)
    text = re.sub(r"\[URL\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df = train_df.with_columns(pl.col("text").map_elements(clean_text).alias("text"))
dev_df = dev_df.with_columns(pl.col("text").map_elements(clean_text).alias("text"))
test_df = test_df.with_columns(pl.col("text").map_elements(clean_text).alias("text"))

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_batch(batch):
    return tokenizer(
        batch["text"].to_list(), padding="max_length", truncation=True, max_length=128
    )


train_encodings = tokenize_batch(train_df)
dev_encodings = tokenize_batch(dev_df)
test_encodings = tokenize_batch(test_df)


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


train_labels = train_df["label"].to_list()
dev_labels = dev_df["label"].to_list()
test_labels = test_df["label"].to_list()

train_dataset = SexismDataset(train_encodings, train_labels)
dev_dataset = SexismDataset(dev_encodings, dev_labels)
test_dataset = SexismDataset(test_encodings, test_labels)

print(len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

batch = next(iter(train_loader))

print(batch["input_ids"].shape)
print(batch["labels"].shape)

"""
Training data:
  not sexist: 10,602 examples (76%)
  sexist:      3,398 examples (24%)
That's why we are using the balanced weight formula
"""

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

save_file(
    {
        "input_ids": torch.tensor(train_encodings["input_ids"]),
        "attention_mask": torch.tensor(train_encodings["attention_mask"]),
        "labels": torch.tensor(train_labels),
    },
    _base / "models" / "train_data.safetensors",
)

save_file(
    {
        "input_ids": torch.tensor(dev_encodings["input_ids"]),
        "attention_mask": torch.tensor(dev_encodings["attention_mask"]),
        "labels": torch.tensor(dev_labels),
    },
    _base / "models" / "dev_data.safetensors",
)

save_file(
    {
        "input_ids": torch.tensor(test_encodings["input_ids"]),
        "attention_mask": torch.tensor(test_encodings["attention_mask"]),
        "labels": torch.tensor(test_labels),
    },
    _base / "models" / "test_data.safetensors",
)

torch.save(
    {
        "class_weights": class_weights,
        "max_length": 128,
        "batch_size": 16,
    },
    _base / "models" / "config.pt",
)

print("Data saved with safetensors")
