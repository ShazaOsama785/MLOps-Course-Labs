from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
with open("model/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Input schema
class PredictionInput(BaseModel):
    features: list

@app.get("/")
def home():
    logger.info("Home endpoint hit")
    return {"message": "Welcome to the SVM Churn Prediction API"}

@app.get("/health")
def health():
    logger.info("Health check passed")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        logger.info(f"Prediction made: {prediction}")
        return {"prediction": int(prediction)}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
