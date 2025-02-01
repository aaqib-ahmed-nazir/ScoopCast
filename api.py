from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI()

# Define the list of feature columns used during training.
feature_columns = [
    "temp", "feels_like", "temp_min", "temp_max",
    "humidity", "dew_point",
    "wind_speed", "wind_deg",
    "clouds_all", "visibility",
    "rain_1h", "rain_3h", "snow_1h", "snow_3h"
]

# Define the input model for each day's forecast including all required fields.
class DayForecast(BaseModel):
    date: str  # date in 'YYYY-MM-DD' format
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    humidity: float
    dew_point: float
    wind_speed: float
    wind_deg: float
    clouds_all: float  # or int, depending on your data
    visibility: float
    rain_1h: float
    rain_3h: float
    snow_1h: float
    snow_3h: float

class ForecastRequest(BaseModel):
    forecast: List[DayForecast]

# Load the trained model.
try:
    model = joblib.load('ice_cream_sales_model.pkl')
except Exception as e:
    raise HTTPException(status_code=500, detail="Model loading failed: " + str(e))

@app.post("/predict")
def predict_sales(request: ForecastRequest):
    # Convert the forecast JSON into a DataFrame.
    data = pd.DataFrame([day.dict() for day in request.forecast])
    
    # Select the features in the same order as used during training.
    X_new = data[feature_columns]
    
    # Predict sales for each flavor (each row corresponds to a day)
    try:
        preds = model.predict(X_new)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))
    
    # Define flavor names manually (adjust these names as per your model training output).
    flavor_names = [
        "Americana",
        "Cheesecake de Frambuesa",
        "Chocolate con Almendras",
        "Crema Oreo",
        "Dulce de Leche Granizado",
        "Maracuyá"
    ]
    
    # Create a JSON-friendly response containing predictions for each day.
    response = []
    for i, prediction in enumerate(preds):
        day_result = {"date": data.iloc[i]["date"]}
        # Map each flavor to its predicted sales.
        for flavor, pred in zip(flavor_names, prediction):
            day_result[flavor] = pred
        response.append(day_result)
    
    return {"predictions": response}

# To run the API, use the command: uvicorn api:app --reload