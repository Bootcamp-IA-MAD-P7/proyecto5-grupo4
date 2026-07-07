import sys

import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import polars as pl

from pathlib import Path

edos = pl.read_csv("data/raw/edos_labelled_aggregated.csv")

sexism = pl.read_csv("data/raw/call_me_sexist.csv")

edos_binary = edos.select(

    [

        pl.col("text"),

        pl.when(pl.col("label_sexist") == "sexist")

        .then(True)

        .otherwise(False)

        .alias("sexist"),

        pl.col("label_category"),

        pl.col("label_vector"),

        pl.lit("edos").alias("source"),

    ]

)

sexism_binary = sexism.select(

    [

        pl.col("text"),

        pl.col("sexist"),

        pl.lit(None).alias("label_category").cast(pl.Utf8),

        pl.lit(None).alias("label_vector").cast(pl.Utf8),

        pl.col("dataset").alias("source"),

        pl.col("toxicity"),

    ]

)

merged = pl.concat([edos_binary, sexism_binary], how="diagonal")

Path("data/processed").mkdir(parents=True, exist_ok=True)

merged.write_parquet("data/processed/merged_edos_sexism.parquet")

print(f"Saved to data/processed/merged_edos_sexism.parquet")

print(f"Merged shape: {merged.shape}")

print(f"By source:")

print(merged["source"].value_counts().sort("count", descending=True))

print(f"Sexist ratio:")

print(merged["sexist"].value_counts().sort("count", descending=True))
