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
    "day_of_week",
    "temp",
    "feels_like",
    "temp_min",
    "temp_max",
    "humidity",
    "dew_point",
    "wind_speed",
    "wind_deg",
    "clouds_all",
    "visibility",
    "rain_1h",
    "rain_3h",
    "snow_1h",
    "snow_3h",
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
    model = joblib.load(
        "/Users/aaqibnazir/Documents/work/ScoopCast/model/ice_cream_sales_model.pkl"
    )
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
        "Maracuyá",
        "Banana con Dulce",
        "Banana sin tacc x 16 unidades",
        "Banana sin tacc x 27 unidades",
        "Bombón Rocher",
        "Candy Americana",
        "Candy Chocolate",
        "ChocoKing",
        "Chocolate",
        "Chocolate 80 sin tacc x 27 unidades",
        "Chocolate 80%",
        "Chocolate Alfonsina",
        "Chocolate Bariloche",
        "Chocolate Blanco",
        "Chocolate Dietetico",
        "Chocolate Patagonia",
        "Chocolate Placer",
        "Chocotorta",
        "Crema de Almendras",
        "Crema de Cerezas y Maraschino",
        "Dulce de Leche",
        "Dulce de Leche 1972",
        "Dulce de Leche Split",
        "Dulce de leche Placer",
        "Dulce de leche sin tacc x 16 unidades",
        "Dulce de leche sin tacc x 27 unidades",
        "Flan con dulce",
        "Fruti Cream",
        "Frutilla Dietetico",
        "Frutilla a la crema",
        "Frutilla al agua",
        "Frutilla al agua sin tacc x 16 unidades",
        "Frutilla al agua sin tacc x 27 unidades",
        "Granizado",
        "Inter Miami CF",
        "Limón (a la crema)",
        "Limón 3 Rios",
        "Limón Negro",
        "Limón al agua",
        "Mantecol",
        "Mascarpone con Frutos del Bosque",
        "Maxi Chocotorta",
        "Maxi Oreo",
        "Maxi Tiramisú",
        "Menta Granizada",
        "Mousse de Chocolate",
        "Mousse de Maracuya",
        "Paleta Bombon x 18 unidades",
        "Paleta Choco 80 x 18 unidades",
        "Paleta Intermiami",
        "Paleta Intermiami x 18 unidades",
        "Paleta Nutella x 18 unidades",
        "Paleta Oreo x 18 unidades",
        "Pan Integral (16) x Unid",
        "Pan x Unid (16)",
        "Pistaccio",
        "Porciones Brownie (20) x Unid",
        "Postre Bombon x 3 Unid",
        "Postre Cheesecake x Unid",
        "Postre Chocotorta x Unid",
        "Postre Rocher x Unid",
        "Postre Tiramisú x Unid",
        "Postre Torta Oreo x Unid",
        "Roll Oreo",
        "Roll Premium (Mascarpone Almendras Choco 80%)",
        "Roll clasico (frutilla-dulce-choco)",
        "Super Sambayón",
        "Super nute",
        "Tiramisú Italiano",
        "Torta Chocotorta",
        "Torta Oreo",
        "Tramontana",
        "Vainilla",
    ]

    # Creating the response
    response = []
    for i, prediction in enumerate(preds):
        day_result = {"date": data.iloc[i]["date"]}
        for flavor, pred in zip(flavor_names, prediction):
            day_result[flavor] = pred
        response.append(day_result)

    return {"predictions": response}
