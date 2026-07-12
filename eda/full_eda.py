import polars as pl
import altair as alt
import pyarrow as pa
import matplotlib.pyplot as plt
import seaborn as sns
import vegafusion
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from textblob import TextBlob


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
# %% [markdown]
# ## Inspección de variables
#
# Revisión de la estructura del dataset combinado: dimensiones, tipos de dato por columna y muestra de filas.

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
# Distribution of label_sexist
sexist_counts = df["label_sexist"].value_counts().sort("count", descending=True)
print("noLabel distribution:")
print(sexist_counts)

# %%
df_pd["label_category"] = df_pd["label_category"].replace("none", "not sexist")
df_pd["label_vector"] = df_pd["label_vector"].replace("none", "not sexist")

CATEGORIES = [
    "not sexist",
    "1. threats, plans to harm and incitement",
    "2. derogation",
    "3. animosity",
    "4. prejudiced discussions",
]

category_dropdown = alt.binding_select(
    options=[None] + CATEGORIES, labels=["All"] + CATEGORIES, name="Category: "
)
category_select = alt.selection_point(fields=["label_category"], bind=category_dropdown)

cat_counts = df_pd["label_category"].value_counts().reindex(CATEGORIES).reset_index()
print("\nCategory distribution: ")
print(cat_counts)
# %%
# Distribution of label_vector
vector_counts = df["label_vector"].value_counts().sort("count", descending=True)
print("\n=== Vector distribution ===")
print(vector_counts)

# %%
# Text length analysis
df_pd["text_length"] = df_pd["text"].str.len()

text_len_by_cat = (
    alt.Chart(df_pd)
    .mark_boxplot()
    .encode(
        x=alt.X("label_category:N", title="Category"),
        y=alt.Y("text_length:Q", title="Text Length (chars)"),
        color=alt.Color("label_category:N", scale=alt.Scale(scheme="rainbow")),
    )
    .properties(title="Text Length by Category", width=500)
)
text_len_by_cat.display()

# %%
# Text length by sexist label
text_len_by_sexist = (
    alt.Chart(df_pd)
    .mark_boxplot()
    .encode(
        x=alt.X("label_sexist:N", title="Sexist"),
        y=alt.Y("text_length:Q", title="Text Length (chars)"),
        color=alt.Color("label_sexist:N", scale=alt.Scale(scheme="rainbow")),
    )
    .properties(title="Text Length by Sexist Label", width=300)
)
text_len_by_sexist.display()

# %%
# Text length vs Word count. Interactive with category selection
df_pd["text_length"] = df_pd["text"].str.len()
df_pd["word_count"] = df_pd["text"].apply(lambda x: len(x.split()))

category_select = alt.selection_point(fields=["label_category"], bind="legend")

cat_dd1 = alt.binding_select(
    options=[None] + CATEGORIES, labels=["All"] + CATEGORIES, name="Category: "
)
cat_sel1 = alt.selection_point(fields=["label_category"], bind=cat_dd1)

scatter_len_words = (
    alt.Chart(df_pd)
    .mark_circle(opacity=0.6, size=20)
    .encode(
        x=alt.X("text_length:Q", title="Text Length (chars)"),
        y=alt.Y("word_count:Q", title="Word Count"),
        color=alt.Color(
            "label_category:N", title="Category", scale=alt.Scale(scheme="rainbow")
        ),
        opacity=alt.condition(cat_sel1, alt.value(0.7), alt.value(0.05)),
        tooltip=["text", "label_sexist", "label_category", "label_vector"],
    )
    .add_params(cat_sel1)
    .properties(title="Text Length vs Word Count", width=600, height=400)
    .interactive()
)
scatter_len_words.display()

# %%
# Category breakdown within sexist tweets
sexist_only = df_pd[df_pd["label_sexist"] == "sexist"]
cat_pie = (
    alt.Chart(sexist_only)
    .mark_arc()
    .encode(
        theta=alt.Theta("count()", title="Count"),
        color=alt.Color("label_category:N", title="Category"),
    )
    .properties(title="Category Distribution (Sexist Only)")
)
cat_pie.display()

# %%
print(f"\nTotal rows: {len(df)}")
print(f"Sexist: {len(df_sexist)} ({len(df_sexist) / len(df) * 100:.1f}%)")
print(
    f"Not sexist: {len(df) - len(df_sexist)} ({(len(df) - len(df_sexist)) / len(df) * 100:.1f}%)"
)

"""
PCA on TF-IDF.
TF-IDF (Term Frequency–Inverse Document Frequency) converts each tweet into a numerical vector based on how important each word is to that tweet relative to the whole dataset.
PCA (Principal Component Analysis) takes those high-dimensional vectors (1000 dimensions in this case) and compresses them down to 2 dimensions so you can plot them on a scatter.
In short: each tweet becomes a point on a 2D plot where nearby points have similar word patterns. Outliers (isolated points far from the clusters) are structurally different from the rest — could be mislabeled, edge cases, or unusual writing styles.
"""
# %%
# Which sexist tweets are similar? Taking only sexist tweets for better visualization
sexist_pd = df_pd[df_pd["label_sexist"] == "sexist"].copy()

sexist_pd = df_pd[df_pd["label_sexist"] == "sexist"].copy()

vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
tfidf = vectorizer.fit_transform(sexist_pd["text"])

pca = PCA(n_components=2)
coords = pca.fit_transform(tfidf.toarray())
sexist_pd["pc1"] = coords[:, 0]
sexist_pd["pc2"] = coords[:, 1]

feature_names = vectorizer.get_feature_names_out()
top_pc1 = [feature_names[i] for i in pca.components_[0].argsort()[-5:]]
top_pc2 = [feature_names[i] for i in pca.components_[1].argsort()[-5:]]

cat_dd = alt.binding_select(
    options=[None] + CATEGORIES[1:],
    labels=["All sexist"] + CATEGORIES[1:],
    name="Category: ",
)
cat_sel = alt.selection_point(fields=["label_category"], bind=cat_dd)

scatter = (
    alt.Chart(sexist_pd)
    .mark_circle(opacity=0.6, size=25)
    .encode(
        x=alt.X("pc1:Q", title="Tone & style of the tweet"),
        y=alt.Y("pc2:Q", title="Topic & vocabulary choice"),
        color=alt.Color(
            "label_category:N", title="Category", scale=alt.Scale(scheme="rainbow")
        ),
        opacity=alt.condition(cat_sel, alt.value(0.7), alt.value(0.05)),
        tooltip=["text", "label_category", "label_vector"],
    )
    .add_params(cat_sel)
    .properties(
        title="Sexist tweets grouped by wording (close = similar)",
        width=600,
        height=400,
    )
    .interactive()
)
scatter.display()

print(f"Tone & style most common words: {top_pc1}")
print(f"Topic & vocabulary most common words: {top_pc2}")

# %%
sexist_labels = df_pd["label_sexist"].unique().tolist()
sexist_dropdown = alt.binding_select(
    options=[None] + sexist_labels, labels=["All"] + sexist_labels, name="Filter: "
)
sexist_select = alt.selection_point(fields=["label_sexist"], bind=sexist_dropdown)

scatter_pca_sexist = (
    alt.Chart(df_pd)
    .mark_circle(opacity=0.6, size=20)
    .encode(
        x=alt.X("pca_x:Q", title="Tone & style →"),
        y=alt.Y("pca_y:Q", title="Topic & vocabulary →"),
        color=alt.Color(
            "label_sexist:N", title="Sexist", scale=alt.Scale(scheme="rainbow")
        ),
        opacity=alt.condition(sexist_select, alt.value(0.7), alt.value(0.05)),
        tooltip=["text", "label_sexist", "label_category", "label_vector"],
    )
    .add_params(sexist_select)
    .properties(
        title="Sexist vs non-sexist tweets (close = similar wording)",
        width=600,
        height=400,
    )
    .interactive()
)
scatter_pca_sexist.display()

# %%
# Known slurs in "not sexist" tweets
slurs = ["bitch", "slut", "whore", "cunt", "hoe", "thot"]
slur_pattern = "|".join(slurs)
slurs_in_not_sexist = df_pd[
    (df_pd["label_sexist"] == "not sexist")
    & (df_pd["text"].str.lower().str.contains(slur_pattern))
]
print(f"\n=== Slurs in 'not sexist' tweets: {len(slurs_in_not_sexist)} rows ===")
print(slurs_in_not_sexist[["text", "label_category"]].head(10))
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
# Distribución de la variable objetivo `sexist`, ya unificada como booleana en el dataset combinado.

# %%
print(df["sexist"].value_counts())
# %%

# %% [markdown]
# ## Variable derivada: cantidad de palabras
#
# Se crea `word_count` a partir de `text`, contando la cantidad de palabras por comentario, para tener una variable numérica adicional a cruzar en la matriz de correlación.

# %%
df = df.with_columns(pl.col("text").str.split(" ").list.len().alias("word_count"))
print(df.select(["text", "word_count"]).head(5))

# %% [markdown]
# ## Matriz de correlación
#
# Correlación de Pearson entre las variables numéricas del dataset: `sexist` (target, tratado como 0/1), `toxicity` y `word_count`.

# %%
corr_df = df.select(["sexist", "toxicity", "word_count"]).to_pandas()
corr_matrix = corr_df.corr()
print(corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de correlación")
plt.tight_layout()
plt.show()
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
