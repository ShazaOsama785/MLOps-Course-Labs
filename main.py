from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import logging

# ------------------------
# Step 1: Setup logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Step 2: Load model & preprocessor
# ------------------------
try:
    with open("model/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")

    with open("model/column_transformer.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Loading failed: {e}")
    raise

# ------------------------
# Step 3: Create FastAPI app
# ------------------------
app = FastAPI()

# ------------------------
# Step 4: Input Schema
# ------------------------
class PredictionInput(BaseModel):
    features: list

# ------------------------
# Step 5: Routes
# ------------------------
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
        logger.info(f"Received data: {data.features}")
        
        # Define the original input feature names
        input_columns = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ]

        # Convert list to DataFrame with column names
        input_df = pd.DataFrame([data.features], columns=input_columns)

        # Transform input
        transformed_input = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(transformed_input)[0]
        logger.info(f"Prediction made: {prediction}")
        return {"prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
