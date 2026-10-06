import logging
from typing import Dict, List
import sqlite3
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ml_models = {}
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- MODEL REGISTRY (Simulation) ---
    if not os.path.exists("model.pkl"):
        logger.info("model.pkl not found locally. Fetching from Model Registry (S3/GCS/MLflow)...")
        time.sleep(1) # Simulate network download
        # boto3.client('s3').download_file('my-ml-models', 'iris/prod/model.pkl', 'model.pkl')
        logger.info("Download complete.")
    # -----------------------------------

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

# --- DATA DRIFT LOGGING ---
def log_prediction_to_db(features: dict, species: str, confidence: float):
    """Asynchronously logs predictions to SQLite to monitor for data drift."""
    try:
        conn = sqlite3.connect("drift_logs.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                      sepal_length REAL, sepal_width REAL, 
                      petal_length REAL, petal_width REAL, 
                      species TEXT, confidence REAL)''')
        c.execute('''INSERT INTO predictions (sepal_length, sepal_width, petal_length, petal_width, species, confidence)
                     VALUES (?, ?, ?, ?, ?, ?)''', 
                  (features['sepal_length'], features['sepal_width'], features['petal_length'], features['petal_width'], species, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Drift logging failed: {e}")
# --------------------------

class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0.1, lt=15.0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0.1, lt=15.0, description="Sepal width in cm")
    petal_length: float = Field(gt=0.1, lt=15.0, description="Petal length in cm")
    petal_width: float = Field(gt=0.1, lt=15.0, description="Petal width in cm")

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
async def predict(features: IrisFeatures, background_tasks: BackgroundTasks):
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
        background_tasks.add_task(log_prediction_to_db, features.model_dump(), species, confidence)
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
