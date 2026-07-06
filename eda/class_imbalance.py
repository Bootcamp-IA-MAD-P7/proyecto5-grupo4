import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

df = pl.read_csv("data/raw/edos_labelled_aggregated.csv")

counts = df["label_sexist"].value_counts().sort("count", descending=True)
print("Class distribution:")
print(counts)

total = len(df)
for row in counts.iter_rows(named=True):
    print(f"  {row['label_sexist']}: {row['count']} ({row['count']/total*100:.1f}%)")

fig, ax = plt.subplots()
ax.bar(["not sexist", "sexist"], [
    counts.filter(pl.col("label_sexist") == "not sexist")["count"][0],
    counts.filter(pl.col("label_sexist") == "sexist")["count"][0],
])
ax.set_title("Class Imbalance")
ax.set_ylabel("Count")
plt.savefig("eda/class_imbalance.png")
print("Saved to eda/class_imbalance.png")
