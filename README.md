# Sexism Detector — NLP Binary Classification

API de detección de sexismo en texto usando un ensemble de **DistilBERT + Logistic Regression**. Entrenado sobre el dataset EDOS (SemEval-2023 Task 10) con ~14,000 muestras etiquetadas.

**Live API:** https://sexism-detector-1000036994845.europe-west1.run.app

---

## Tabla de contenidos

- [Quickstart](#quickstart)
- [Arquitectura del proyecto](#arquitectura-del-proyecto)
- [Dataset](#dataset)
- [Modelos](#modelos)
  - [Logistic Regression + TF-IDF](#logistic-regression--tf-idf)
  - [DistilBERT](#distilbert)
  - [Ensemble](#ensemble)
- [API](#api)
- [Docker](#docker)
- [Deploy (Google Cloud Run)](#deploy-google-cloud-run)
- [Resultados](#resultados)
- [Equipo](#equipo)

---

## Quickstart

```bash
# Probar la API desplegada
curl -X POST https://sexism-detector-1000036994845.europe-west1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Women belong in the kitchen"}'

# Respuesta
{"text": "Women belong in the kitchen", "label": "sexist", "confidence": 0.77}
```

---

## Arquitectura del proyecto

```
proyecto5-grupo4/
├── backend/
│   └── main.py                  # API FastAPI (endpoints /health y /predict)
├── scripts/
│   └── predict.py               # Lógica de predicción (ensemble)
├── models/
│   ├── distilbert/
│   │   ├── data_pipeline.py     # Preprocesamiento y tokenización
│   │   └── train_model.py       # Entrenamiento con Focal Loss
│   ├── logistic_regression/
│   │   ├── data_pipeline.py     # Limpieza de texto y splits
│   │   ├── train_model.py       # TF-IDF + Logistic Regression
│   │   ├── model.joblib          # Modelo entrenado
│   │   └── vectorizer.joblib     # Vectorizador TF-IDF
│   └── ensemble/
│       ├── trained_ensemble.py  # Búsqueda de pesos óptimos
│       └── weights.json          # Pesos: 0.60 DB / 0.40 LR
├── data/
│   ├── raw/                     # Dataset original EDOS
│   └── processed/               # Splits en parquet
├── eda/                         # Análisis exploratorio
├── Dockerfile                   # Contenedor para deploy
├── pyproject.toml               # Dependencias (uv)
└── uv.lock                      # Versiones exactas
```

---

## Dataset

**EDOS (Explainable Detection of Online Sexism)** — SemEval-2023 Task 10

| Split | Muestras | Not sexist | Sexist |
|-------|----------|-----------|--------|
| Train | 14,000 | 76% | 24% |
| Dev | 2,000 | 76% | 24% |
| Test | 4,000 | 76% | 24% |

**Desbalance de clases:** 76/24 — abordado con class weights balanceados y Focal Loss.

**Preprocesamiento:**
- Lowercase
- Eliminación de placeholders `[USER]`, `[URL]`
- Normalización de espacios

---

## Modelos

### Logistic Regression + TF-IDF

Modelo clásico que captura patrones léxicos (slurs, frecuencia de palabras clave).

- **Vectorización:** TF-IDF con configuración por defecto (unigrams)
- **Modelo:** Logistic Regression con `class_weight="balanced"`, `max_iter=1000`
- **Resultado:** F1 (sexist) = 0.63, ROC-AUC = 0.847

Se probó tuning de hiperparámetros (C, ngram_range, min_df) pero el modelo por defecto generalizó mejor — el tuning sobreajustó al dev set.

### DistilBERT

Fine-tuned **DistilBERT-base-uncased** (66.9M parámetros) para clasificación binaria.

- **Tokenización:** max 128 tokens
- **Training:** 5 epochs, batch size 16, learning rate 2e-5, AdamW
- **Loss:** Focal Loss (gamma=1.0) con class weights balanceados
- **Threshold:** 0.5 (óptimo tras tuning con precision_recall_curve)
- **GPU:** Entrenado en Google Colab con T4 GPU (~6 min)

**Modelo publicado en HuggingFace:** [Anahia/sexism-detector-distilbert](https://huggingface.co/Anahia/sexism-detector-distilbert)

#### Experimentos con Focal Loss

| Gamma | F1 (sexist) | Precision | Recall | Accuracy |
|:-----:|:-----------:|:---------:|:------:|:--------:|
| Baseline (CE) | 0.72 | 0.68 | 0.76 | 85% |
| **1.0** | **0.72** | **0.69** | **0.76** | **86%** |
| 1.5 | 0.71 | 0.66 | 0.78 | 85% |
| 2.0 | 0.59 | 0.43 | 0.94 | 82% |

### Ensemble

Promedio ponderado de las probabilidades de ambos modelos.

**Pesos óptimos:** 60% DistilBERT + 40% Logistic Regression (buscados en 21 pasos sobre dev set).

**¿Por qué ensemble?**
- LR detecta bien casos obvios (slurs explícitos) pero falla en sutilezas
- DistilBERT captura contexto y sarcasmo pero genera más falsos positivos
- Sus errores son complementarios — combinarlos mejora ambos

---

## Resultados finales (Ensemble en test set)

```
              precision    recall  f1-score   support
  not sexist       0.92      0.90      0.91      3030
      sexist       0.70      0.76      0.73       970

    accuracy                           0.86      4000
   macro avg       0.81      0.83      0.82      4000
```

| Métrica | LR | DistilBERT | Ensemble |
|---|---|---|---|
| F1 (sexist) | 0.63 | 0.72 | **0.73** |
| Precision | — | 0.69 | **0.70** |
| Recall | — | 0.76 | **0.76** |
| Accuracy | — | 86% | **86%** |

---

## API

FastAPI con dos endpoints:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predicción de sexismo |

**Request:**
```json
POST /predict
{"text": "texto a analizar"}
```

**Response:**
```json
{
  "text": "texto a analizar",
  "label": "sexist",
  "confidence": 0.77
}
```

---

## Docker

```bash
# Construir imagen
docker build -t sexism-detector .

# Ejecutar en local
docker run -d -p 8080:8080 --name sexism-api sexism-detector

# Probar
curl http://localhost:8080/health
```

El Dockerfile usa `uv` para instalar dependencias y expone el puerto 8080.

---

## Deploy (Google Cloud Run)

La API está desplegada en Google Cloud Run con 2GB de RAM.

```bash
# Requisitos: gcloud CLI instalado y autenticado
gcloud auth login
gcloud config set project sexism-detector-api

# Habilitar servicios
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Desplegar
gcloud run deploy sexism-detector \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

**URL de producción:** https://sexism-detector-1000036994845.europe-west1.run.app

**Flujo del deploy:**
```
Código local → Cloud Build (construye Docker) → Artifact Registry (guarda imagen) → Cloud Run (ejecuta) → URL pública
```

---

## Tech Stack

- **Python 3.12**
- **FastAPI** — API REST
- **PyTorch + HuggingFace Transformers** — DistilBERT
- **scikit-learn** — Logistic Regression + TF-IDF
- **Polars** — procesamiento de datos
- **Docker** — contenedorización
- **uv** — gestión de dependencias
- **Google Cloud Run** — deploy en producción

---

## Frontend

Pendiente de integración. El frontend se conecta a la API mediante fetch POST a:

```javascript
fetch("https://sexism-detector-1000036994845.europe-west1.run.app/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "texto a analizar" })
})
```

---

## Equipo

Proyecto 5 — Grupo 4 | Bootcamp IA MAD P7
