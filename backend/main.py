# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for frontend communication
origins = [
    "http://localhost:3000",  # React default port
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load crime data once at startup
try:
    crime_data = pd.read_csv('data/crimes_by_state.csv')
except FileNotFoundError:
    raise Exception("Crime data file not found. Ensure 'crimes_by_state.csv' is in the 'data/' directory.")

# Load the trained Random Forest model once at startup
try:
    model = joblib.load('models/random_forest_model_for_house_price_prediction.joblib')
except FileNotFoundError:
    raise Exception("Model file not found. Ensure 'random_forest_model_for_house_price_prediction.joblib' is in the 'models/' directory.")

# Pydantic models for request and response

class PredictionRequest(BaseModel):
    state: str
    bedrooms: int
    baths: int
    salary_per_year: Optional[float] = None  # Optional for affordability

class PredictionResponse(BaseModel):
    predicted_price: float
    yearly_income: Optional[float] = None
    monthly_payment: Optional[float] = None
    affordability_status: Optional[str] = None

# Affordability calculation function
def calculate_affordability(price, annual_salary, interest_rate=0.06, years=30):
    monthly_interest_rate = interest_rate / 12
    total_payments = years * 12
    monthly_payment = price * (monthly_interest_rate * (1 + monthly_interest_rate) ** total_payments) / (
        (1 + monthly_interest_rate) ** total_payments - 1)
    affordable_payment = (annual_salary / 12) * 0.30
    return monthly_payment <= affordable_payment, monthly_payment

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_house_price(request: PredictionRequest):
    state = request.state
    bedrooms = request.bedrooms
    baths = request.baths
    salary_per_year = request.salary_per_year

    # Retrieve crime statistics for the state
    crime_stats = crime_data.loc[crime_data['state'].str.lower() == state.lower()]
    if crime_stats.empty:
        raise HTTPException(status_code=404, detail="State not found in dataset.")

    property_crime = crime_stats['property_per_100_000'].values[0]
    violence_crime = crime_stats['violence_per_100_000'].values[0]

    # Create input DataFrame for prediction
    input_data = pd.DataFrame({
        'beds': [bedrooms],
        'baths': [baths],
        'property_per_100_000': [property_crime],
        'violence_per_100_000': [violence_crime],
        'state': [state]
    })

    # Predict using Random Forest
    try:
        predicted_price_rf = model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    response = PredictionResponse(predicted_price=predicted_price_rf)

    if salary_per_year is not None:
        is_affordable, monthly_payment = calculate_affordability(predicted_price_rf, salary_per_year)
        response.yearly_income = salary_per_year
        response.monthly_payment = monthly_payment
        response.affordability_status = "Affordable" if is_affordable else "Not Affordable"

    return response
