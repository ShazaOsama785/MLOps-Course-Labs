import pytest
import pickle
import numpy as np
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

# Load preprocessor and model from disk
@pytest.fixture(scope="module")
def load_artifacts():
    with open("model/column_transformer.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("model/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return preprocessor, model

# ✅ Test 1: Check loading works
def test_load_artifacts(load_artifacts):
    preprocessor, model = load_artifacts
    assert preprocessor is not None
    assert model is not None

# ✅ Test 2: Check prediction works
def test_model_prediction(load_artifacts):
    preprocessor, model = load_artifacts

    sample = pd.DataFrame([{
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }])

    transformed = preprocessor.transform(sample)
    prediction = model.predict(transformed)
    assert prediction[0] in [0, 1]

# ✅ Test 3: Check input shape after transform
def test_preprocessor_shape(load_artifacts):
    preprocessor, _ = load_artifacts

    sample = pd.DataFrame([{
        "CreditScore": 650,
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 30,
        "Tenure": 5,
        "Balance": 70000,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 40000
    }])

    transformed = preprocessor.transform(sample)
    assert transformed.shape[0] == 1  # one row
    assert transformed.shape[1] > 0   # has features
   
