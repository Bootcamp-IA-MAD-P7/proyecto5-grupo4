import polars as pl
from pathlib import Path

edos = pl.read_csv(r"data/raw/edos_labelled_aggregated.csv")
sexism = pl.read_csv(r"data/raw/call_me_sexist.csv")

sexism = sexism.rename({"id": "sexism_id"})

merged = edos.join(sexism, on="text", how="full").drop("text_right")

Path("data/processed").mkdir(parents=True, exist_ok=True)
merged.write_parquet("data/processed/merged_edos_sexism.parquet")

print("✓ Saved to data/processed/merged_edos_sexism.parquet")
print(f"✓ Merged shape: {merged.shape}")
