from fastapi import FastAPI
from pydantic import BaseModel
from scripts.predict import predict_sexism

app = FastAPI(title="Sexism Detector API")


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
