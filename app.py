import logging
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Loading model.pkl...")
    try:
        ml_models["model"] = joblib.load("model.pkl")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
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

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    logger.info(f"Received prediction request: {features}")
    if "model" not in ml_models:
        logger.error("Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Prepare the data as a DataFrame to preserve feature names if needed
        data = pd.DataFrame([features.model_dump()])
        prediction = ml_models["model"].predict(data)
        species = prediction[0]
        logger.info(f"Prediction successful: {species}")
        return PredictionResponse(species=species)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static directory to serve the frontend (must be at the end so it doesn't override API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
