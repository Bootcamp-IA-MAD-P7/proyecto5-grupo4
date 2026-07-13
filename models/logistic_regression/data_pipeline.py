# =============================================================================
# Sexism Detection — Logistic Regression with TF-IDF
# Data pipeline: load, filter splits, encode labels, clean text, persist
# EDOS dataset (SemEval-2023 Task 10)
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import re
from pathlib import Path

import polars as pl


# -----------------------------------------------------------------------------
# Base path resolution
# -----------------------------------------------------------------------------
try:
    _base = Path(__file__).resolve().parent.parent.parent
except NameError:
    _base = Path(os.getcwd())


# -----------------------------------------------------------------------------
# 1. Data loading
# -----------------------------------------------------------------------------
df = pl.read_csv(_base / "data" / "raw" / "edos_labelled_aggregated.csv")


# -----------------------------------------------------------------------------
# 2. Split filtering and label encoding
# The original EDOS split is preserved for comparability with the source paper.
# -----------------------------------------------------------------------------
train_df = df.filter(pl.col("split") == "train")
dev_df = df.filter(pl.col("split") == "dev")
test_df = df.filter(pl.col("split") == "test")

train_df = train_df.with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
dev_df = dev_df.with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
test_df = test_df.with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)


# -----------------------------------------------------------------------------
# 3. Text cleaning
# Identical to the DistilBERT pipeline so both models process text the same
# way. Note: .lower() runs before the [USER]/[URL] regexes, so those patterns
# never match — this is a known bug, kept deliberately for comparability
# between models. Fixing it is pending a team decision.
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[USER\]", "", text)
    text = re.sub(r"\[URL\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df = train_df.with_columns(
    pl.col("text").map_elements(clean_text, return_dtype=pl.String).alias("text")
)
dev_df = dev_df.with_columns(
    pl.col("text").map_elements(clean_text, return_dtype=pl.String).alias("text")
)
test_df = test_df.with_columns(
    pl.col("text").map_elements(clean_text, return_dtype=pl.String).alias("text")
)


# -----------------------------------------------------------------------------
# 4. Persist processed splits
# -----------------------------------------------------------------------------
processed_dir = _base / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

train_df.write_parquet(processed_dir / "train.parquet")
dev_df.write_parquet(processed_dir / "dev.parquet")
test_df.write_parquet(processed_dir / "test.parquet")


# -----------------------------------------------------------------------------
# 5. Verification prints
# -----------------------------------------------------------------------------
print(f"Train: {train_df.shape}")
print(f"Dev:   {dev_df.shape}")
print(f"Test:  {test_df.shape}")

print(train_df["label"].value_counts())
print(dev_df["label"].value_counts())
print(test_df["label"].value_counts())

print("Processed splits saved to", processed_dir)
