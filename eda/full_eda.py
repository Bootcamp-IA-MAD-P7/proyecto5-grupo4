import polars as pl
import altair as alt
import pyarrow as pa
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
alt.data_transformers.enable("vegafusion")
plt.style.use("seaborn-v0_8-whitegrid")
try:
    _base = Path(__file__).resolve().parent.parent
except NameError:
    _base = Path(os.getcwd()).parent
df = pl.read_parquet(_base / "data" / "processed" / "merged_edos_sexism.parquet")
df_pd = df.to_pandas()

# %%
# %% [markdown]
# ## Inspección de variables
#
# Revisión de la estructura del dataset combinado: dimensiones, tipos de dato por columna y muestra de filas.

# %%
print(df.shape)
print(df.schema)
df.head(5)

# %% [markdown]
# ## Valores nulos
#
# Conteo de valores nulos por columna, en cantidad y porcentaje sobre el total de filas.

# %%
nulls = df.null_count()
total = df.height
for col in df.columns:
    n = nulls[col][0]
    pct = round(100 * n / total, 2)
    print(f"{col:<20} {n:>6}  ({pct}%)")

# %% [markdown]
# ## Análisis de la variable objetivo
#
# Distribución de la variable objetivo en cada dataset de origen (`label_sexist` en EDOS, `sexist` en CMSB) y unificación en una única columna `target` booleana.

# %%
print(df["label_sexist"].value_counts())
print(df["sexist"].value_counts())

df = df.with_columns(
    pl.when(pl.col("label_sexist").is_not_null())
      .then(pl.col("label_sexist") == "sexist")
      .otherwise(pl.col("sexist"))
      .alias("target")
)

print(df["target"].value_counts())
print("Nulos en target:", df["target"].null_count())

df_pd = df.to_pandas()
# %%
print(df["label_sexist"].value_counts())
print(df["sexist"].value_counts())

df = df.with_columns(
    pl.when(pl.col("label_sexist").is_not_null())
      .then(pl.col("label_sexist") == "sexist")
      .otherwise(pl.col("sexist"))
      .alias("target")
)

print(df["target"].value_counts())
print("Nulos en target:", df["target"].null_count())

df_pd = df.to_pandas()

# %%
"""
Code for saving this file as a notebook.
# It Goes at the end of the file.
"""
# %%
if __name__ == "__main__":
    import jupytext as jpt

    _py = Path(__file__)
    _nb = _py.with_suffix(".ipynb")
    if not _nb.exists():
        jpt.write(jpt.read(_py), _nb, fmt="ipynb")
        print(f"Created {_nb.name}")
    else:
        jpt.write(jpt.read(_py), _nb, fmt="ipynb")
        print(f"Synced {_nb.name}")

# %%
