"""
Day 20: FastAPI Web Service for Machine Learning Inference
Core Topics: FastAPI, Uvicorn, Joblib Model Loading, Pydantic Request Validation
"""

import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# ---- 1. Initialize FastAPI App ----
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML Engineering Sprint - Day 20 Production API using FastAPI & Scikit-Learn",
    version="1.0.0"
)

# ---- 2. Load the Saved Model Pipeline from Disk ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "saved_models", "churn_pipeline.joblib")
churn_model = joblib.load(model_path)                    # load exported binary joblib pipeline

# ---- 3. Define Request Data Schema using Pydantic ----
# Ensures incoming client JSON payload matches expected types and features
class CustomerData(BaseModel):
    tenure_months: int                                   # customer tenure in months
    monthly_charges: float                               # monthly bill amount
    support_calls: int                                   # support calls count

# ---- 4. Define Root Health Check Endpoint ----
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to the Machine Learning Churn Prediction API!"
    }

# ---- 5. Define Real-Time Prediction POST Endpoint ----
@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert incoming Pydantic request body into DataFrame/2D array format expected by pipeline
    input_features = [[
        data.tenure_months,
        data.monthly_charges,
        data.support_calls
    ]]

    # Generate prediction using loaded joblib pipeline
    prediction = churn_model.predict(input_features)[0]
    prediction_proba = churn_model.predict_proba(input_features)[0].max()

    # Formulate business response
    result_label = "Churned (1)" if prediction == 1 else "Stayed (0)"

    return {
        "prediction_code": int(prediction),
        "prediction_label": result_label,
        "confidence_score": round(float(prediction_proba) * 100, 2)
    }