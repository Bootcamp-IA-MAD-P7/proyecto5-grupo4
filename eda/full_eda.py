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
# Which sexist tweets are similar?
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Taking only sexist tweets for better visualization
sexist_pd = df_pd[df_pd["label_sexist"] == "sexist"].copy()

sexist_pd = df_pd[df_pd["label_sexist"] == "sexist"].copy()

vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
tfidf = vectorizer.fit_transform(sexist_pd["text"])

pca = PCA(n_components=2)
coords = pca.fit_transform(tfidf.toarray())
sexist_pd["pc1"] = coords[:, 0]
sexist_pd["pc2"] = coords[:, 1]

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
