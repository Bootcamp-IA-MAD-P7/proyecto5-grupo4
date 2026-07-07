# %%
import polars as pl
import altair as alt
import pyarrow as pa
import matplotlib.pyplot as plt
import seaborn as sns
import vegafusion
import os
from pathlib import Path

alt.data_transformers.enable("vegafusion")
plt.style.use("seaborn-v0_8-whitegrid")

try:
    _base = Path(__file__).resolve().parent.parent
except NameError:
    _base = Path(os.getcwd()).parent
df = pl.read_csv(_base / "data" / "raw" / "edos_labelled_aggregated.csv")
df_pd = df.to_pandas().reset_index()
df_sexist = df.filter(pl.col("label_sexist") == "sexist")
df_sexist_pd = df_sexist.to_pandas().reset_index()

# %% 
# Nombre de las columnas
print(df.columns)

# %% 
# Datos de las columnas
print(df.head())

# %% 
"""
# Histograma 1 : Distribución general de los datos
"""

# %% 
plt.figure(figsize=(10, 6))
sns.countplot(data=df_pd, x="label_sexist", palette="viridis")
plt.title("Distribución de mensajes: Sexistas vs No Sexistas")
plt.xlabel("Categoría")
plt.ylabel("Cantidad de mensajes")
plt.show()

# %% 
"""
# Histograma 2: Distribución por tipo de etiqueta detallada
"""

# %% 
plt.figure(figsize=(12, 6))
sns.countplot(data=df_pd, y="label_category", hue="label_sexist", palette="magma")
plt.title("Desglose detallado de categorías de sexismo")
plt.xlabel("Cantidad de mensajes")
plt.ylabel("Categoría específica")
plt.show()

# %% 
"""
# Histograma 3: Balance de datos entre Train y Dev
"""

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df_pd, x="split", hue="label_sexist", palette="coolwarm")
plt.title("Balance de etiquetas sexistas en cada set (Train vs Dev)")
plt.xlabel("Conjunto de datos")
plt.ylabel("Cantidad de mensajes")
plt.show()

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
