import os
from pathlib import Path
import re
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.utils.class_weight import compute_class_weight


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

print(type(train_encodings))
print(train_encodings["input_ids"][0][:10])
