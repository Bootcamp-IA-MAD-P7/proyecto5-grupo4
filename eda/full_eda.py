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
df = pl.read_parquet(_base / "data" / "processed" / "merged_edos_sexism.parquet")
df_pd = df.to_pandas().reset_index()

"""
ML-17. Outlier analysis
"""
# %%

box = (
    alt.Chart(df_pd)
    .mark_boxplot(color="steelblue")
    .encode(y=alt.Y("toxicity", title="Toxicity Score"))
    .properties(title="Boxplot - Toxicity", width=200)
)

hist = (
    alt.Chart(df_pd)
    .mark_bar()
    .encode(
        x=alt.X("toxicity", bin=True, title="Toxicity Score"),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("toxicity", bin=True, scale=alt.Scale(scheme="rainbow")),
    )
    .properties(title="Distribution - Toxicity", width=400)
)

scatter = (
    alt.Chart(df_pd)
    .mark_circle(opacity=0.4, size=30)
    .encode(
        x=alt.X("index:Q", title="Index"),
        y=alt.Y("toxicity", title="Toxicity Score"),
        color=alt.Color("toxicity", scale=alt.Scale(scheme="rainbow")),
    )
    .properties(title="Scatter - Toxicity", width=400)
)

(box | hist).resolve_scale(color="independent").display()
scatter.display()

# %%
scatter = (
    alt.Chart(df_pd)
    .mark_rect()
    .encode(
        x=alt.X("index:Q", bin=alt.Bin(maxbins=50)),
        y=alt.Y("toxicity", bin=alt.Bin(maxbins=30)),
        color=alt.Color("count()", scale=alt.Scale(scheme="rainbow")),
    )
).show()
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
