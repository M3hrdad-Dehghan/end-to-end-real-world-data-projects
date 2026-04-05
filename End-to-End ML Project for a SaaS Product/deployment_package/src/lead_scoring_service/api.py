from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
DEFAULT_TRACKING_URI = "file:///" + (PROJECT_ROOT / "experiments" / "mlruns").as_posix().lstrip("/")
DEFAULT_MODEL_URI = "models:/LeadScoringService/latest"

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)

    mlflow.set_tracking_uri(tracking_uri)
    _model = mlflow.pyfunc.load_model(model_uri)

    yield

    _model = None


app = FastAPI(title="Lead Scoring API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> list[dict[str, Any]]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        frame = pd.DataFrame(request.rows)
        float_cols = ["avg_session_duration_min", "avg_ticket_resolution_hours"]
        for col in float_cols:
            if col in frame.columns:
                frame[col] = frame[col].astype(float)
        result = _model.predict(frame)
        return result.to_dict(orient="records")
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


