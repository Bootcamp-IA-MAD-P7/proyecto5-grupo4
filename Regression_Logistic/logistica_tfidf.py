# =============================================================================
# Sexism Detection — Logistic Regression with TF-IDF
# EDOS dataset (SemEval-2023 Task 10)
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import re
from pathlib import Path

import joblib
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Match the preprocessing applied in the DistilBERT pipeline."""
    text = text.lower()
    text = re.sub(r"\[USER\]", "", text)
    text = re.sub(r"\[URL\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------------
# 1. Data loading
# -----------------------------------------------------------------------------
_base = Path.cwd()
if _base.name == "Regression_Logistic":
    _base = _base.parent

df = pl.read_csv(_base / "data" / "raw" / "edos_labelled_aggregated.csv")


# -----------------------------------------------------------------------------
# 2. Split filtering and label encoding
# The original EDOS split is preserved for comparability with the source paper.
# -----------------------------------------------------------------------------
train_df = df.filter(pl.col("split") == "train").with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
dev_df = df.filter(pl.col("split") == "dev").with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)
test_df = df.filter(pl.col("split") == "test").with_columns(
    pl.when(pl.col("label_sexist") == "sexist").then(1).otherwise(0).alias("label")
)


# -----------------------------------------------------------------------------
# 3. Feature / target separation
# Text is cleaned to match the DistilBERT pipeline preprocessing.
# -----------------------------------------------------------------------------
X_train = train_df["text"].map_elements(clean_text, return_dtype=pl.String)
y_train = train_df["label"]

X_dev = dev_df["text"].map_elements(clean_text, return_dtype=pl.String)
y_dev = dev_df["label"]

X_test = test_df["text"].map_elements(clean_text, return_dtype=pl.String)
y_test = test_df["label"]

print(X_train.shape, y_train.shape)
print(X_dev.shape, y_dev.shape)
print(X_test.shape, y_test.shape)

# -----------------------------------------------------------------------------
# 4. TF-IDF vectorization
# The vocabulary is fitted on the training set only, then applied to dev and
# test. Fitting on unseen data would introduce data leakage.
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)
print(X_dev_tfidf.shape)
print(X_test_tfidf.shape)


# -----------------------------------------------------------------------------
# 5. Baseline model
# class_weight="balanced" compensates the 76/24 class imbalance confirmed in EDA.
# -----------------------------------------------------------------------------
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

print("Model trained")


# -----------------------------------------------------------------------------
# 6. Baseline evaluation on test set
# Accuracy alone is misleading under class imbalance: precision, recall, F1 and
# ROC-AUC are reported per class.
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred, target_names=["not sexist", "sexist"]))
print(confusion_matrix(y_test, y_pred))

y_proba = model.predict_proba(X_test_tfidf)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# -----------------------------------------------------------------------------
# 7. Hyperparameter tuning — C
# Each candidate is evaluated on the dev set. The test set remains untouched
# until the final evaluation.
# -----------------------------------------------------------------------------
for C in [0.01, 0.1, 1, 10, 100]:
    tuning_model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    tuning_model.fit(X_train_tfidf, y_train)

    y_dev_pred = tuning_model.predict(X_dev_tfidf)
    dev_f1 = f1_score(y_dev, y_dev_pred)

    print(f"C={C}: F1 (sexist) = {dev_f1:.4f}")

# -----------------------------------------------------------------------------
# 8. Hyperparameter tuning — vectorizer
# Changing the vectorizer requires re-fitting it and re-transforming the data,
# not just re-training the model.
# -----------------------------------------------------------------------------
configs = [
    {"ngram_range": (1, 1), "min_df": 1},
    {"ngram_range": (1, 2), "min_df": 3},
    {"ngram_range": (1, 3), "min_df": 2},
    {"ngram_range": (1, 3), "min_df": 5},
    {"ngram_range": (1, 3), "min_df": 2},
    {"ngram_range": (1, 4), "min_df": 2},
    {"ngram_range": (1, 5), "min_df": 2},
]

for config in configs:
    tuning_vectorizer = TfidfVectorizer(**config)

    X_train_tuning = tuning_vectorizer.fit_transform(X_train)
    X_dev_tuning = tuning_vectorizer.transform(X_dev)

    tuning_model = LogisticRegression(C=1, class_weight="balanced", max_iter=1000)
    tuning_model.fit(X_train_tuning, y_train)

    y_dev_pred = tuning_model.predict(X_dev_tuning)
    dev_f1 = f1_score(y_dev, y_dev_pred)

    vocab_size = X_train_tuning.shape[1]
    print(f"{config} → vocab={vocab_size:>7} | F1 (sexist) = {dev_f1:.4f}")

# -----------------------------------------------------------------------------
# 9. Re-tuning C on the winning vectorizer
# -----------------------------------------------------------------------------
best_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)

X_train_best = best_vectorizer.fit_transform(X_train)
X_dev_best = best_vectorizer.transform(X_dev)

for C in [0.1, 0.5, 1, 2, 5, 10]:
    tuning_model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    tuning_model.fit(X_train_best, y_train)

    y_dev_pred = tuning_model.predict(X_dev_best)
    dev_f1 = f1_score(y_dev, y_dev_pred)

    print(f"C={C}: F1 (sexist) = {dev_f1:.4f}")


# -----------------------------------------------------------------------------
# 10. Tuned model — evaluated once on the test set
# Configuration selected via dev-set grid search (sections 7-9).
# Result: underperforms the baseline on test. Kept as evidence of the
# validation-set overfitting phenomenon.
# -----------------------------------------------------------------------------
tuned_model = LogisticRegression(
    C=1, class_weight="balanced", max_iter=1000, random_state=42
)
tuned_model.fit(X_train_best, y_train)

X_test_best = best_vectorizer.transform(X_test)
y_test_pred = tuned_model.predict(X_test_best)
y_test_proba = tuned_model.predict_proba(X_test_best)[:, 1]

print("\n=== TUNED MODEL — TEST SET ===")
print(classification_report(y_test, y_test_pred, target_names=["not sexist", "sexist"]))
print(confusion_matrix(y_test, y_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_test_proba))

# -----------------------------------------------------------------------------
# 11. Save artifacts
# The vectorizer holds the learned vocabulary, the model holds the learned
# weights. Probabilities and labels are saved for the ensemble stage.
# Note: .joblib files execute Python code when loaded. Only load artifacts
# generated by this repository.
# -----------------------------------------------------------------------------
artifacts_dir = _base / "models" / "logistic_regression"
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(vectorizer, artifacts_dir / "vectorizer.joblib")
joblib.dump(model, artifacts_dir / "model.joblib")

dev_proba = model.predict_proba(X_dev_tfidf)[:, 1]
test_proba = model.predict_proba(X_test_tfidf)[:, 1]

np.save(artifacts_dir / "dev_probs.npy", dev_proba)
np.save(artifacts_dir / "test_probs.npy", test_proba)
np.save(artifacts_dir / "dev_labels.npy", y_dev.to_numpy())
np.save(artifacts_dir / "test_labels.npy", y_test.to_numpy())

print("Artifacts saved")