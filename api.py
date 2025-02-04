import joblib
import pandas as pd
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

"""
 - Defining the FastAPI app 
"""
app = FastAPI()

# Defining the feature columns
feature_columns = [
    "day_of_week","temp", "feels_like", "temp_min", "temp_max",
    "humidity", "dew_point",
    "wind_speed", "wind_deg",
    "clouds_all", "visibility",
    "rain_1h", "rain_3h", "snow_1h", "snow_3h"
]

# Defining the request and response models
class DayForecast(BaseModel):
    date: str  
    day_of_week: int
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    humidity: float
    dew_point: float
    wind_speed: float
    wind_deg: float
    clouds_all: float 
    visibility: float
    rain_1h: float
    rain_3h: float
    snow_1h: float
    snow_3h: float

# Defining the request model
class ForecastRequest(BaseModel):
    forecast: List[DayForecast]

# Loading the model
try:
    model = joblib.load('/Users/aaqibnazir/Documents/work/ScoopCast/model/ice_cream_sales_model.pkl')
except Exception as e:
    raise HTTPException(status_code=500, detail="Model loading failed: " + str(e))

# Endpoint for making predictions
@app.post("/predict")
def predict_sales(request: ForecastRequest):
    """
        - Parameters:
            - request: ForecastRequest
                - forecast: List[DayForecast]
                    - date: str
                    - day_of_week: int
                    - temp: float
                    - feels_like: float
                    - temp_min: float
                    - temp_max: float
                    - humidity: float
                    - dew_point: float
                    - wind_speed: float
                    - wind_deg: float
                    - clouds_all: float
                    - visibility: float
                    - rain_1h: float
                    - rain_3h: float
                    - snow_1h: float
                    - snow_3h: float
        - Returns:
            - predictions: List[Dict[str, Union[str, float]]]
                - date: str
                - Americana: float
                - Cheesecake de Frambuesa: float
                - Chocolate con Almendras: float
                - Crema Oreo: float
                - Dulce de Leche Granizado: float
                - Maracuyá: float
                
                
        - Description:
            - This endpoint receives a list of DayForecast objects and returns a list of 
                dictionaries with the predictions for each day.
    """
    
    data = pd.DataFrame([day.dict() for day in request.forecast])
    
    X_new = data[feature_columns]
    
    # Making predictions
    try:
        preds = model.predict(X_new)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))
    
    flavor_names = [
        "Americana",
        "Cheesecake de Frambuesa",
        "Chocolate con Almendras",
        "Crema Oreo",
        "Dulce de Leche Granizado",
        "Maracuyá"
    ]
    
    # Creating the response
    response = []
    for i, prediction in enumerate(preds):
        day_result = {"date": data.iloc[i]["date"]}
        for flavor, pred in zip(flavor_names, prediction):
            day_result[flavor] = pred
        response.append(day_result)
    
    return {"predictions": response}