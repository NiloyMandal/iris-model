import logging
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ml_models = {}
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Loading model.pkl...")
    try:
        ml_models["model"] = joblib.load("model.pkl")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    # Load the reference dataset (used by the frontend scatter plot)
    try:
        df = pd.read_csv("iris.csv").dropna()
        ml_models["samples"] = df[FEATURES + ["species"]].to_dict(orient="records")
        logger.info(f"Loaded {len(ml_models['samples'])} reference samples.")
    except Exception as e:
        logger.warning(f"Could not load iris.csv for /samples: {e}")
        ml_models["samples"] = []
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    species: str
    confidence: float = 1.0
    probabilities: Dict[str, float] = {}

class Sample(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    logger.info(f"Received prediction request: {features}")
    if "model" not in ml_models:
        logger.error("Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Prepare the data as a DataFrame to preserve feature names if needed
        data = pd.DataFrame([features.model_dump()])
        model = ml_models["model"]
        species = str(model.predict(data)[0])

        probabilities: Dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(data)[0]
            probabilities = {str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)}
        confidence = probabilities.get(species, 1.0)

        logger.info(f"Prediction successful: {species} (confidence={confidence:.2f})")
        return PredictionResponse(species=species, confidence=confidence, probabilities=probabilities)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/samples", response_model=List[Sample])
async def samples():
    """Reference iris measurements, used by the frontend to plot the user's flower in context."""
    return ml_models.get("samples", [])

# Mount the static directory to serve the frontend (must be at the end so it doesn't override API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
