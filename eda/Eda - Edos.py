# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown]
# # EDA — EDOS (Explainable Detection of Online Sexism)
#
# ## Origen del dataset
#
# EDOS fue publicado como la **Task 10 de SemEval-2023**, la competición internacional
# de evaluación semántica. Sus autores son **Hannah Rose Kirk, Wenjie Yin, Bertie Vidgen
# y Paul Röttger**, un equipo conjunto de **Rewire** (empresa especializada en IA para
# seguridad online), la **Universidad de Oxford** y **Queen Mary University of London**.
#
# El dataset contiene **20.000 comentarios en inglés** recogidos de **Reddit y Gab**,
# anotados por personas entrenadas con supervisión experta. Su aporte principal frente a
# datasets anteriores es que no se limita a la etiqueta binaria: propone una **taxonomía
# jerárquica de tres niveles** — (1) sexista / no sexista, (2) categoría del sexismo,
# (3) vector específico — lo que permite construir modelos *explicables*, que no solo
# detectan sino que justifican la clasificación.
#


# %% [markdown]
# ## 0. Configuración e importaciones
#
# Stack del proyecto: Polars para manipulación de datos, Altair (+ VegaFusion) para
# gráficos interactivos, matplotlib/seaborn para gráficos estáticos.

# %%
import polars as pl
import altair as alt
import pyarrow as pa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import vegafusion
import os
from pathlib import Path

alt.data_transformers.enable("vegafusion")
plt.style.use("seaborn-v0_8-whitegrid")

# Paleta del proyecto — tonalidades rojo/rosa (de oscuro a claro)
PALETA = ["#590D22", "#800F2F", "#A4133C", "#C9184A", "#FF4D6D",
          "#FF758F", "#FF8FA3", "#FFB3C1", "#FFCCD5"]
CMAP_ROSA = mcolors.LinearSegmentedColormap.from_list(
    "rosas", ["#FFF0F3", "#FF8FA3", "#C9184A", "#590D22"]
)
sns.set_palette(PALETA)

try:
    _base = Path(__file__).resolve().parent.parent
except NameError:
    _base = Path(os.getcwd()).parent

# %% [markdown]
# ## 1. Carga del dataset
#
# Se carga el CSV crudo original de EDOS. Se mantiene una copia en pandas (`df_pd`)
# únicamente como puente para los gráficos de seaborn, que no aceptan Polars.

# %%
df = pl.read_csv(_base / "data" / "raw" / "edos_labelled_aggregated.csv")
df_pd = df.to_pandas()
df.head()

# %% [markdown]
# ## 2. Estructura del dataset
#
# Dimensiones y tipos de dato de cada columna.

# %%
print(f"Filas: {df.shape[0]:,} — Columnas: {df.shape[1]}")
df.schema

# %% [markdown]
# ## 3. Calidad de datos — valores nulos
#
# Conteo de nulos por columna sobre el dataset completo.

# %%
df.null_count()

# %% [markdown]
# ## 4. Calidad de datos — duplicados
#
# Filas completamente duplicadas y duplicados de texto (mismo comentario repetido
# con distinto id, algo posible al recolectar de dos plataformas).

# %%
print(f"Filas duplicadas completas: {df.is_duplicated().sum()}")
print(f"Textos duplicados: {df['text'].is_duplicated().sum()}")

# %% [markdown]
# ## 5. Análisis univariado — variable objetivo `label_sexist`
#
# Distribución de la clase binaria que el modelo deberá predecir.

# %%
conteo_clase = (
    df["label_sexist"]
    .value_counts(sort=True)
    .with_columns((pl.col("count") / pl.col("count").sum() * 100).round(2).alias("pct"))
)
conteo_clase

# %% [markdown]
# ### Visualización — gráfico de anillo interactivo
#
# El anillo muestra la proporción de cada clase; al pasar el cursor se ven los
# valores exactos.

# %%
donut = (
    alt.Chart(conteo_clase.to_pandas())
    .mark_arc(innerRadius=70, cornerRadius=6)
    .encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color(
            "label_sexist:N",
            scale=alt.Scale(domain=["not sexist", "sexist"],
                            range=["#FFB3C1", "#A4133C"]),
            legend=alt.Legend(title="Clase"),
        ),
        tooltip=["label_sexist:N", "count:Q", "pct:Q"],
    )
    .properties(width=380, height=320, title="Distribución de la variable objetivo")
)
donut

# %% [markdown]
# # Conclusión — Distribución de `label_sexist`
#
# El anillo muestra un desbalance de clases marcado: **75.73% not sexist** frente a **24.27% sexist**. 
# Esto confirma la decisión ya tomada de usar `class_weight="balanced"` en el modelado, y descarta el accuracy 
# como métrica suficiente — precision, recall, F1 y matriz de confusión por clase serán necesarios para evaluar 
# correctamente el desempeño sobre la clase minoritaria.

# %% [markdown]
# ## 6. Análisis univariado — `label_category`
#
# Categorías de sexismo (segundo nivel de la taxonomía), calculadas solo sobre el
# subconjunto etiquetado como sexista.

# %%
df_sexist = df.filter(pl.col("label_sexist") == "sexist")

conteo_cat = (
    df_sexist["label_category"]
    .value_counts(sort=True)
    .with_columns((pl.col("count") / pl.col("count").sum() * 100).round(2).alias("pct"))
)
conteo_cat

# %% [markdown]
# ### Visualización — lollipop chart
#
# Cada "chupetín" representa una categoría; la longitud del palo es el conteo.

# %%
cat_pd = conteo_cat.to_pandas().sort_values("count")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hlines(y=cat_pd["label_category"], xmin=0, xmax=cat_pd["count"],
          color="#FF8FA3", linewidth=2.5)
ax.plot(cat_pd["count"], cat_pd["label_category"], "o",
        markersize=11, color="#800F2F")
for _, row in cat_pd.iterrows():
    ax.text(row["count"] + 40, row["label_category"],
            f'{row["count"]:,} ({row["pct"]}%)', va="center",
            fontsize=9, color="#590D22")
ax.set_xlim(0, cat_pd["count"].max() * 1.22)
ax.set_title("Categorías de sexismo (solo clase sexista)", fontsize=13, color="#590D22")
ax.set_xlabel("Cantidad de comentarios")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Conclusión — categorías de sexismo
#
# Dentro de la clase sexista, la distribución de categorías es desigual:
# *derogation* concentra casi la mitad de los casos (46.8%), seguida por
# *animosity* (34.3%), mientras que *prejudiced discussions* (9.8%) y
# *threats* (9.1%) son minoritarias.
#
# La clase minoritaria del problema binario no es homogénea, sino una mezcla
# desigual de cuatro subtipos. Aunque el objetivo del proyecto es la
# clasificación binaria, esto sugiere evaluar el modelo también por categoría:
# un buen desempeño global podría esconder puntos ciegos sistemáticos en los
# subtipos menos representados, especialmente *threats*.

# %% [markdown]
# ## 7. Análisis univariado — `label_vector`
#
# Tercer nivel de la taxonomía: el vector específico dentro de cada categoría.
# Se ordenan de mayor a menor frecuencia (solo clase sexista).

# %%
conteo_vec = df_sexist["label_vector"].value_counts(sort=True)
conteo_vec

# %% [markdown]
# ### Visualización — barras horizontales con degradé
#
# Interactivo: tooltip con el conteo exacto de cada vector.

# %%
barras_vec = (
    alt.Chart(conteo_vec.to_pandas())
    .mark_bar(cornerRadiusEnd=5)
    .encode(
        x=alt.X("count:Q", title="Cantidad"),
        y=alt.Y("label_vector:N", sort="-x", title=None),
        color=alt.Color("count:Q", scale=alt.Scale(range=["#FFCCD5", "#590D22"]),
                        legend=None),
        tooltip=["label_vector:N", "count:Q"],
    )
    .properties(width=560, height=340, title="Vectores de sexismo (nivel 3 de la taxonomía)")
)
barras_vec

# %% [markdown]
# # Conclusión — Distribución de `label_vector` (nivel 3 de la taxonomía)
#
# Las tres subcategorías más frecuentes — **descriptive attacks** (21.1%), **aggressive and emotive attacks** (19.8%) 
# y **casual use of gendered slurs, profanities, and insults** (18.7%) — concentran el **59.6%** de todo el contenido 
# sexista, y corresponden a las categorías de nivel 2 ya identificadas como dominantes (*derogation* y *animosity*, 
# sección 6). En el extremo opuesto, **condescending explanations or unwelcome advice** (1.4%, 68 casos) y **threats 
# of harm** (1.6%, 80 casos) son las subcategorías más escasas del dataset completo.
#
# Esta granularidad expone un riesgo que la sección 6 no mostraba, con menos de 100 ejemplos en tres subcategorías, 
# un split estratificado a este nivel es poco viable. Si el modelo llega a evaluarse por `label_vector`, hay que 
# considerar agrupar las clases más pequeñas o aceptar que su desempeño individual no será estadísticamente confiable.

# %% [markdown]
# ## 8. Análisis univariado — `split`
#
# Verificación del split original del paper (train / dev / test) y su proporción,
# tanto global como dentro de cada clase.

# %%
conteo_split = (
    df.group_by("split", "label_sexist")
    .len()
    .sort("split", "label_sexist")
)
conteo_split

# %% [markdown]
# ### Visualización — barras apiladas al 100%
#
# Permite comprobar que el desbalance de clases se mantiene estable en los tres splits
# (requisito para que la evaluación sea justa).

# %%
split_chart = (
    alt.Chart(conteo_split.to_pandas())
    .mark_bar(cornerRadius=4)
    .encode(
        x=alt.X("split:N", sort=["train", "dev", "test"], title="Split"),
        y=alt.Y("len:Q", stack="normalize", title="Proporción",
                axis=alt.Axis(format="%")),
        color=alt.Color("label_sexist:N",
                        scale=alt.Scale(domain=["not sexist", "sexist"],
                                        range=["#FFB3C1", "#A4133C"]),
                        legend=alt.Legend(title="Clase")),
        tooltip=["split:N", "label_sexist:N", "len:Q"],
    )
    .properties(width=340, height=300, title="Balance de clases por split")
)
split_chart

# %% [markdown]
# # Conclusión — Balance de clases por split
#
# EDOS se distribuye con una columna `split` predefinida que divide el dataset en train, dev y test; 
# siguiendo el paper original, respetamos esa partición en lugar de generar una propia. Este análisis 
# verifica que esa decisión es segura.
#
# La proporción de contenido sexista se mantiene estable en los tres subconjuntos: **24.27%** en train, 
# **24.30%** en dev y **24.25%** en test. El split oficial ya viene estratificado respecto a `label_sexist`, 
# por lo que conserva el desbalance global en cada partición.
#
# Esto respalda respetar el split original sin re-particionar ni re-estratificar, y garantiza que las métricas 
# obtenidas sobre test serán representativas de la distribución vista en entrenamiento. Al ser el desbalance 
# idéntico en todas las particiones, el tratamiento con `class_weight="balanced"` se aplica de forma consistente 
# en todo el flujo.

# %% [markdown]
# ## 9. Ingeniería de variables numéricas a partir del texto
#
# El dataset solo tiene columnas de texto y etiquetas, por lo que para poder hacer
# análisis de distribuciones y correlaciones se derivan variables numéricas de cada
# comentario: longitud, cantidad de palabras, longitud media de palabra, proporción de
# mayúsculas, exclamaciones, interrogaciones y menciones anonimizadas ([USER], [URL]).

# %%
df_feat = df.with_columns(
    pl.col("text").str.len_chars().alias("n_chars"),
    pl.col("text").str.split(" ").list.len().alias("n_words"),
    pl.col("text").str.count_matches(r"[A-Z]").alias("n_upper"),
    pl.col("text").str.count_matches(r"!").alias("n_exclam"),
    pl.col("text").str.count_matches(r"\?").alias("n_question"),
    pl.col("text").str.count_matches(r"\[USER\]").alias("n_user"),
    pl.col("text").str.count_matches(r"\[URL\]").alias("n_url"),
    (pl.col("label_sexist") == "sexist").cast(pl.Int8).alias("es_sexista"),
).with_columns(
    (pl.col("n_chars") / pl.col("n_words")).round(2).alias("largo_medio_palabra"),
    (pl.col("n_upper") / pl.col("n_chars")).round(4).alias("prop_mayusculas"),
)

df_feat.select(
    "n_chars", "n_words", "largo_medio_palabra", "prop_mayusculas",
    "n_exclam", "n_question", "n_user", "n_url"
).describe()

# %% [markdown]
# # Conclusión — Estadística descriptiva de las features de texto
#
# Las 8 features derivadas del texto no presentan valores nulos, confirmando que la construcción fue correcta 
# sobre las 20.000 filas. El comentario típico ronda los **127 caracteres** y **23 palabras**, con media y mediana 
# muy próximas en ambas, lo que indica una distribución de longitud razonablemente simétrica. El máximo exacto de 
# **250 caracteres** refleja el truncado propio del dataset EDOS, no un valor atípico.
#
# Se detectan varias señales a revisar antes del modelado. `largo_medio_palabra` alcanza un máximo de **99** frente 
# a una media de 5.5, valor imposible en lenguaje natural que apunta a URLs o cadenas sin espacios mal capturadas por 
# la feature; conviene inspeccionar esos casos. `prop_mayusculas` llega a **0.92**, revelando comentarios casi 
# íntegramente en mayúsculas. Las variables de conteo (`n_exclam`, `n_question`, `n_user`, `n_url`) están fuertemente 
# sesgadas a cero —P75 = 0 en todas— con colas largas, un patrón esperable en texto de redes donde la mayoría de 
# mensajes no usa estos elementos y una minoría los concentra.
#
# Estas distribuciones sesgadas y los outliers de construcción condicionan el tratamiento posterior, escalado y 
# posible recorte o transformación, si estas features se incorporan al modelo.
# %% [markdown]
# ## 10. Distribución de features — longitud del comentario por clase
#
# Histograma interactivo superpuesto: la distribución de longitud (en caracteres)
# de comentarios sexistas frente a no sexistas. Pre-agregado en Polars para que el
# gráfico sea liviano.

# %%
hist_len = (
    df_feat
    .with_columns((pl.col("n_chars") // 20 * 20).alias("bin_chars"))
    .group_by("bin_chars", "label_sexist")
    .len()
    .sort("bin_chars")
)

hist_chart = (
    alt.Chart(hist_len.to_pandas())
    .mark_area(opacity=0.65, interpolate="monotone", line=True)
    .encode(
        x=alt.X("bin_chars:Q", title="Longitud del comentario (caracteres)"),
        y=alt.Y("len:Q", title="Cantidad de comentarios"),
        color=alt.Color("label_sexist:N",
                        scale=alt.Scale(domain=["not sexist", "sexist"],
                                        range=["#FF8FA3", "#800F2F"]),
                        legend=alt.Legend(title="Clase")),
        tooltip=["bin_chars:Q", "label_sexist:N", "len:Q"],
    )
    .properties(width=640, height=320,
                title="Distribución de longitud del texto por clase")
)
hist_chart

# %% [markdown]
# # Conclusión — Distribución de longitud del texto por clase
#
# Comparando la forma de ambas distribuciones (normalizadas por el total de cada clase, dado el desbalance), 
# los comentarios **no sexistas** concentran su masa en textos más cortos, con pico alrededor de los **60–80 
# caracteres**, mientras que los **sexistas** desplazan su pico hacia los **100–120 caracteres** y presentan una 
# cola derecha más gruesa: proporcionalmente hay más contenido sexista en el tramo largo (180–250 caracteres).
#
# El corrimiento sugiere que la longitud del texto aporta una señal débil —el contenido sexista tiende a ser algo 
# más extenso—, pero el solapamiento entre ambas curvas es casi total, por lo que `n_chars` por sí sola no separa 
# las clases. Puede sumar como feature auxiliar dentro de un conjunto mayor, pero no es un predictor discriminante 
# por sí mismo.

# %% [markdown]
# ## 11. Distribución de features — violines por clase
#
# Los violines muestran la forma completa de la distribución (no solo la mediana)
# de cantidad de palabras y proporción de mayúsculas, separadas por clase.

# %%
feat_pd = df_feat.select(
    "label_sexist", "n_words", "prop_mayusculas", "n_chars", "n_exclam"
).to_pandas()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.violinplot(data=feat_pd, x="label_sexist", y="n_words", hue="label_sexist",
               palette=["#FFB3C1", "#A4133C"], legend=False, ax=axes[0])
axes[0].set_title("Cantidad de palabras por clase", color="#590D22")
axes[0].set_xlabel("")
sns.violinplot(data=feat_pd, x="label_sexist", y="prop_mayusculas", hue="label_sexist",
               palette=["#FFB3C1", "#A4133C"], legend=False, ax=axes[1])
axes[1].set_title("Proporción de mayúsculas por clase", color="#590D22")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Conclusión — Distribución de `n_words` y `prop_mayusculas` por clase
#
# El violin plot permite comparar la forma completa de ambas distribuciones —no solo sus cuartiles—, apropiado 
# dado el sesgo marcado de estas features detectado en la estadística descriptiva.
#
# En **cantidad de palabras**, la clase sexista muestra una mediana y un cuerpo ligeramente desplazados hacia 
# arriba: el contenido sexista tiende a ser algo más extenso, confirmando el mismo patrón observado en la longitud 
# en caracteres. No obstante, las distribuciones se solapan casi por completo, por lo que la señal es débil.
#
# En **proporción de mayúsculas**, ambas distribuciones se concentran fuertemente cerca de cero con colas largas y 
# similares. Contra la hipótesis de que el contenido sexista emplea más mayúsculas (tono de "grito"), no se observa 
# diferencia relevante entre clases; si acaso, la cola de la clase no sexista es levemente mayor.
#
# Ninguna de las dos features separa las clases por sí sola. `n_words` puede aportar señal marginal dentro de un 
# conjunto mayor; `prop_mayusculas` aparece como poco informativa para la clasificación binaria.

# %% [markdown]
# ## 12. Análisis bivariado — longitud del texto según categoría de sexismo
#
# Cruce entre una variable numérica derivada (palabras) y la categoría (nivel 2),
# solo dentro de la clase sexista. Permite ver si algún tipo de sexismo tiende a
# expresarse en textos más largos o más cortos.

# %%
cat_words_pd = (
    df_feat
    .filter(pl.col("label_sexist") == "sexist")
    .select("label_category", "n_words")
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4.5))
orden = cat_words_pd.groupby("label_category")["n_words"].median().sort_values().index
sns.boxplot(data=cat_words_pd, y="label_category", x="n_words", order=orden,
            hue="label_category", palette=["#FFCCD5", "#FF8FA3", "#FF4D6D", "#A4133C"],
            legend=False, ax=ax)
ax.set_title("Palabras por comentario según categoría de sexismo", color="#590D22")
ax.set_ylabel("")
ax.set_xlabel("Cantidad de palabras")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Conclusión — Cantidad de palabras según categoría de sexismo
#
# El boxplot permite comparar la longitud típica entre las cuatro categorías de sexismo de un vistazo, priorizando 
# mediana y dispersión sobre la forma completa. Se observa un gradiente claro: las **amenazas** (*threats, plans to 
# harm and incitement*) son los comentarios más cortos —coherente con su naturaleza breve y directa—, mientras que 
# las **prejudiced discussions** son los más largos, lo esperable en contenido que argumenta o justifica un prejuicio 
# en lugar de atacar de forma inmediata. *Animosity* y *derogation* se ubican en un rango intermedio.
#
# Las diferencias entre medianas son reales pero moderadas, y los rangos de las cuatro categorías se solapan de forma 
# considerable. La longitud del texto guarda relación con el *tipo* de sexismo, pero no lo determina; puede aportar 
# como señal contextual débil, no como discriminador entre subtipos.

# %% [markdown]
# ## 13. Análisis bivariado — menciones anonimizadas por clase
#
# Proporción de comentarios que contienen al menos un [USER] o [URL], por clase.
# Interactivo con tooltips.

# %%
menciones = (
    df_feat
    .group_by("label_sexist")
    .agg(
        (pl.col("n_user") > 0).mean().round(4).alias("con [USER]"),
        (pl.col("n_url") > 0).mean().round(4).alias("con [URL]"),
        (pl.col("n_exclam") > 0).mean().round(4).alias("con exclamación"),
        (pl.col("n_question") > 0).mean().round(4).alias("con interrogación"),
    )
    .unpivot(index="label_sexist", variable_name="marcador", value_name="proporcion")
)

menciones_chart = (
    alt.Chart(menciones.to_pandas())
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("proporcion:Q", axis=alt.Axis(format="%"), title="Proporción de comentarios"),
        y=alt.Y("marcador:N", title=None),
        yOffset="label_sexist:N",
        color=alt.Color("label_sexist:N",
                        scale=alt.Scale(domain=["not sexist", "sexist"],
                                        range=["#FFB3C1", "#A4133C"]),
                        legend=alt.Legend(title="Clase")),
        tooltip=["marcador:N", "label_sexist:N",
                 alt.Tooltip("proporcion:Q", format=".1%")],
    )
    .properties(width=520, height=280,
                title="Marcadores del texto por clase")
)
menciones_chart

# %% [markdown]
# ## 14. Análisis multivariado — matriz de correlación
#
# Correlación de Pearson entre todas las variables numéricas derivadas y la variable
# objetivo codificada (`es_sexista`: 1 = sexista, 0 = no sexista). La última fila/columna
# es la más relevante: qué features se relacionan más con la clase.

# %%
cols_num = ["n_chars", "n_words", "largo_medio_palabra", "prop_mayusculas",
            "n_exclam", "n_question", "n_user", "n_url", "es_sexista"]

corr = df_feat.select(cols_num).corr()
corr

# %% [markdown]
# # Conclusión — Marcadores del texto por clase
#
# Contra la hipótesis de que el contenido sexista concentra más signos de énfasis o interpelación, los cuatro 
# marcadores analizados son **más frecuentes en la clase no sexista**. La brecha mayor está en `[URL]` (13.8% vs 
# 5.6%, más del doble) y en las menciones `[USER]` (5.5% vs 2.7%), mientras que exclamación e interrogación muestran 
# diferencias pequeñas entre clases.
#
# La lectura es que los comentarios no sexistas tienden a ser más conversacionales —comparten enlaces, mencionan a 
# otros usuarios, preguntan—, en tanto que el contenido sexista resulta más autocontenido, con afirmaciones o ataques 
# que no citan fuentes ni interpelan a un interlocutor concreto. Una feature más frecuente en la clase negativa es 
# igualmente informativa para el modelo, por lo que `presencia de [URL]` y `presencia de [USER]` emergen como los 
# marcadores con mayor poder de señal, aunque moderado; los signos de puntuación aportan poco por sí solos.

# %% [markdown]
# ### Visualización — heatmap de correlación
#
# Máscara triangular para no duplicar información; degradé rosa (claro = sin relación,
# oscuro = correlación fuerte, en valor absoluto).

# %%
import numpy as np

corr_np = corr.to_numpy()
mask = np.triu(np.ones_like(corr_np, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_np, mask=mask, annot=True, fmt=".2f",
    cmap=CMAP_ROSA, vmin=-1, vmax=1,
    xticklabels=cols_num, yticklabels=cols_num,
    linewidths=0.6, linecolor="white",
    cbar_kws={"label": "Correlación de Pearson"}, ax=ax,
)
ax.set_title("Matriz de correlación — features derivadas y clase", color="#590D22", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Conclusión — Matriz de correlación
#
# El análisis multivariado arroja dos lecturas. En cuanto a **redundancia entre features**, `n_chars` y `n_words` 
# están correlacionadas casi perfectamente (**0.97**), esperable ya que miden lo mismo (extensión del texto); conviene 
# conservar solo una de las dos para el modelado. El resto de los pares se mantiene por debajo de 0.30, sin 
# colinealidad problemática.
#
# En cuanto al **poder predictivo**, ninguna de las features numéricas derivadas correlaciona de forma relevante con 
# la clase objetivo: todos los coeficientes con `es_sexista` se ubican en torno a ±0.10 o menos, con `n_url` y 
# `prop_mayusculas` (ambas -0.10) como las de mayor magnitud, y negativas —coherente con que esos marcadores son 
# menos frecuentes en el contenido sexista, según secciones anteriores—.
#
# Este es el hallazgo central del EDA, la señal del sexismo no reside en las propiedades superficiales del texto 
# (longitud, mayúsculas, signos, menciones) sino en su **contenido léxico**. El resultado respalda la estrategia de 
# modelado basada en vectorización del texto (CountVectorizer / TF-IDF sobre las palabras), relegando estas features 
# metadata a un rol complementario, si es que se incorporan.

# %% [markdown]
# ## 15. Análisis multivariado — features cruzadas por clase y categoría
#
# Dispersión interactiva de longitud frente a proporción de mayúsculas, coloreada por
# categoría (muestra aleatoria de la clase sexista para mantener el gráfico legible).
# Se puede hacer zoom y ver cada punto con tooltip.

# %%
muestra = (
    df_feat
    .filter(pl.col("label_sexist") == "sexist")
    .sample(n=1500, seed=42)
    .select("n_words", "prop_mayusculas", "label_category")
)

scatter = (
    alt.Chart(muestra.to_pandas())
    .mark_circle(size=45, opacity=0.55)
    .encode(
        x=alt.X("n_words:Q", title="Cantidad de palabras"),
        y=alt.Y("prop_mayusculas:Q", title="Proporción de mayúsculas"),
        color=alt.Color("label_category:N",
                        scale=alt.Scale(range=["#590D22", "#C9184A", "#FF4D6D", "#FFB3C1"]),
                        legend=alt.Legend(title="Categoría")),
        tooltip=["n_words:Q", "prop_mayusculas:Q", "label_category:N"],
    )
    .interactive()
    .properties(width=620, height=380,
                title="Longitud vs. mayúsculas por categoría (muestra, clase sexista)")
)
scatter

# %% [markdown]
# # Conclusión — `n_words` vs `prop_mayusculas` por categoría
#
# El scatter proyecta las dos features numéricas principales con la categoría de sexismo codificada en color, para 
# detectar si algún subtipo ocupa una región diferenciada del plano. El resultado es una **nube única sin estructura 
# de clusters**, las cuatro categorías se superponen por completo, `prop_mayusculas` se concentra cerca de cero en 
# todas ellas y `n_words` se distribuye en todo el rango sin que ninguna categoría domine un tramo específico.
#
# Esto confirma en dos dimensiones lo detectado en la matriz de correlación, las propiedades superficiales del texto 
# no separan ni las clases ni los subtipos de sexismo. La distinción reside en el contenido léxico, lo que refuerza 
# la estrategia de vectorización del texto como base del modelado y relega estas features a un rol, a lo sumo, 
# complementario.

# %% [markdown]
# ## 16. Conclusiones
# %% [markdown]
# # 16. Conclusiones
#
# Síntesis de los hallazgos del análisis exploratorio y sus implicaciones para las fases de preprocesamiento
# y modelado.
#
# **Distribución de la variable objetivo.** El dataset presenta un desbalance marcado: 75.73% no sexista frente
# a 24.27% sexista, proporción estable en los tres splits (train 24.27%, dev 24.30%, test 24.25%). El desbalance
# condiciona toda la estrategia de evaluación y justifica el uso de `class_weight="balanced"` ya acordado.
#
# **Categorías y vectores dominantes.** Dentro de la clase sexista, *derogation* (46.8%) y *animosity* (34.3%)
# concentran la mayoría de los casos, frente a *prejudiced discussions* (9.8%) y *threats* (9.1%). A nivel de
# vector (nivel 3), tres subcategorías —descriptive attacks, aggressive and emotive attacks y casual use of
# gendered slurs— reúnen el 59.6% del contenido sexista, mientras tres vectores quedan por debajo de 100 casos.
# La clase minoritaria es internamente heterogénea y desigual.
#
# **Balance por split.** El split oficial de EDOS viene predefinido y estratificado respecto a `label_sexist`,
# por lo que se respeta sin re-particionar; las métricas sobre test serán representativas de la distribución
# de entrenamiento.
#
# **Features derivadas con más señal.** Ninguna de las 8 features numéricas derivadas del texto (longitud,
# palabras, mayúsculas, signos, menciones) correlaciona de forma relevante con la clase: todos los coeficientes
# con la variable objetivo se ubican en torno a ±0.10. Además, `n_chars` y `n_words` son casi redundantes entre
# sí (r = 0.97). La señal del sexismo no reside en las propiedades superficiales del texto.
#
# **Implicaciones para el preprocesamiento y el modelado.** El hallazgo central orienta toda la estrategia:
# la clasificación debe apoyarse en el **contenido léxico**, no en features tabulares. Esto valida el enfoque de
# vectorización del texto (CountVectorizer / TfidfVectorizer) sobre las palabras como base del modelo, relegando
# las features derivadas a un rol, a lo sumo, complementario. El desbalance confirmado exige `class_weight="balanced"`
# y métricas más allá del accuracy (precision, recall, F1, matriz de confusión, ROC-AUC). La heterogeneidad interna
# de la clase sexista sugiere, si el alcance lo permite, evaluar también el desempeño por categoría para detectar
# puntos ciegos en los subtipos menos representados.
