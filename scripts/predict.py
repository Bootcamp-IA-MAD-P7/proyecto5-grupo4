import torch
import joblib
import json
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pathlib import Path


db_model_path = "Anahia/sexism-detector-distilbert"
lr_model_path = Path("./models/logistic_regression/model.joblib")
lr_vectorizer_path = Path("./models/logistic_regression/vectorizer.joblib")
weights_path = Path("./models/ensemble/weights.json")

db_model = DistilBertForSequenceClassification.from_pretrained(db_model_path)
tokenizer = DistilBertTokenizer.from_pretrained(db_model_path)
lr_model = joblib.load(lr_model_path)
lr_vectorizer = joblib.load(lr_vectorizer_path)

with open(weights_path) as f:
    weights = json.load(f)

DB_WEIGHT = weights["distilbert_weight"]
LR_WEIGHT = weights["logistic_regression_weight"]


def predict_sexism(text):
    db_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        db_outputs = db_model(**db_inputs)

    db_probs = torch.nn.functional.softmax(db_outputs.logits, dim=-1)
    db_prob = db_probs[0, 1].item()

    lr_tfidf = lr_vectorizer.transform([text])
    lr_prob = lr_model.predict_proba(lr_tfidf)[0, 1]

    ensemble_prob = DB_WEIGHT * db_prob + LR_WEIGHT * lr_prob

    return {
        "text": text,
        "label": "sexist" if ensemble_prob > 0.5 else "not_sexist",
        "confidence": float(ensemble_prob),
    }


if __name__ == "__main__":
    test_cases = [
        "Women belong in the kitchen.",
        "Great weather today.",
    ]

    for text in test_cases:
        result = predict_sexism(text)
        print(f"Text: {result['text']}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}\n")
