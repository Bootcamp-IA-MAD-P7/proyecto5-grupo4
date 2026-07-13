from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scripts.predict import predict_sexism

app = FastAPI(title="Sexism Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://proyecto5-grupo4.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictInput(BaseModel):
    text: str


class PredictOutput(BaseModel):
    text: str
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOutput)
def predict(input: PredictInput):
    return predict_sexism(input.text)
