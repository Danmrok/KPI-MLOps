from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from .schemas import IrisFeatures, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

state: dict = {"model": None}


@asynccontextmanager
async def lifespan(application: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state["model"] = None


app = FastAPI(
    title="Iris ML API",
    description="REST API для класифікації квіток Iris",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "Iris ML API"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_loaded": state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    x = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    class_id = int(state["model"].predict(x)[0])
    proba = float(state["model"].predict_proba(x)[0, class_id])

    return PredictionResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        probability=round(proba, 4),
    )
