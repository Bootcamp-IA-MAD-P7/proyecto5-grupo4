FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY models/logistic_regression/model.joblib ./models/logistic_regression/model.joblib
COPY models/logistic_regression/vectorizer.joblib ./models/logistic_regression/vectorizer.joblib
COPY models/ensemble/weights.json ./models/ensemble/weights.json

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
