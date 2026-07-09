
from pathlib import Path
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score



_base = Path.cwd()
if _base.name == "Regression_Logistic":
    _base = _base.parent

df = pl.read_csv(_base / "data" / "raw" / "edos_labelled_aggregated.csv")

train_df = df.filter(pl.col("split") == "train").with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
dev_df = df.filter(pl.col("split") == "dev").with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
test_df = df.filter(pl.col("split") == "test").with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)

model = LogisticRegression(class_weight="balanced", max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Model trained")

y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred, target_names=["not sexist", "sexist"]))
print(confusion_matrix(y_test, y_pred))

y_proba = model.predict_proba(X_test_tfidf)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_proba))