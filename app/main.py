from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# Load trained model
model = joblib.load(Path("model/model.pkl"))


app = FastAPI(
    title="MLOps House Price Prediction API",
    version="1.0.0"
)


class HouseFeatures(BaseModel):
    area_sqft: float
    bedrooms: int
    age_years: float
    distance_km: float


@app.get("/")
def home():
    return {
        "message": "MLOps House Price Prediction API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(features: HouseFeatures):

    # Create DataFrame with the same feature names
    # used during model training
    values = pd.DataFrame([{
        "area_sqft": features.area_sqft,
        "bedrooms": features.bedrooms,
        "age_years": features.age_years,
        "distance_km": features.distance_km
    }])

    prediction = float(model.predict(values)[0])

    return {
        "predicted_price_lakh": round(prediction, 2)
    }