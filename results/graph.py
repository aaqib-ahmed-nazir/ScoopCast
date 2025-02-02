import matplotlib.pyplot as plt
import json

# Load the JSON data
data = {
    "forecast": [
        {
            "date": "2025-02-02",
            "temp": 25.0,
            "feels_like": 26.0,
            "temp_min": 23.0,
            "temp_max": 28.0,
        },
        {
            "date": "2025-02-03",
            "temp": 24.0,
            "feels_like": 25.0,
            "temp_min": 22.0,
            "temp_max": 27.0,
        },
        {
            "date": "2025-02-04",
            "temp": 23.0,
            "feels_like": 24.0,
            "temp_min": 21.0,
            "temp_max": 26.0,
        },
        {
            "date": "2025-02-05",
            "temp": 26.0,
            "feels_like": 27.0,
            "temp_min": 24.0,
            "temp_max": 29.0,
        },
        {
            "date": "2025-02-06",
            "temp": 27.0,
            "feels_like": 28.0,
            "temp_min": 25.0,
            "temp_max": 30.0,
        },
    ],
    "predictions": [
        {
            "date": "2025-02-02",
            "Americana": 16.0,
            "Cheesecake de Frambuesa": 11.9,
            "Chocolate con Almendras": 25.7,
            "Crema Oreo": 24.5,
            "Dulce de Leche Granizado": 33.9,
            "Maracuyá": 8.6,
        },
        {
            "date": "2025-02-03",
            "Americana": 15.0,
            "Cheesecake de Frambuesa": 9.1,
            "Chocolate con Almendras": 23.6,
            "Crema Oreo": 21.5,
            "Dulce de Leche Granizado": 30.9,
            "Maracuyá": 6.8,
        },
        {
            "date": "2025-02-04",
            "Americana": 19.6,
            "Cheesecake de Frambuesa": 9.5,
            "Chocolate con Almendras": 25.7,
            "Crema Oreo": 26.6,
            "Dulce de Leche Granizado": 36.6,
            "Maracuyá": 8.5,
        },
        {
            "date": "2025-02-05",
            "Americana": 16.2,
            "Cheesecake de Frambuesa": 10.1,
            "Chocolate con Almendras": 30.1,
            "Crema Oreo": 25.8,
            "Dulce de Leche Granizado": 41.6,
            "Maracuyá": 10.6,
        },
        {
            "date": "2025-02-06",
            "Americana": 20.5,
            "Cheesecake de Frambuesa": 11.6,
            "Chocolate con Almendras": 30.2,
            "Crema Oreo": 28.9,
            "Dulce de Leche Granizado": 50.8,
            "Maracuyá": 12.7,
        },
    ]
}

# Extract forecast data
dates = [item["date"] for item in data["forecast"]]
temps = [item["temp"] for item in data["forecast"]]
feels_like = [item["feels_like"] for item in data["forecast"]]
temp_min = [item["temp_min"] for item in data["forecast"]]
temp_max = [item["temp_max"] for item in data["forecast"]]

# Plot Weather Forecast
plt.figure(figsize=(10, 6))
plt.plot(dates, temps, label="Temperature (°C)", marker="o", linestyle="-", color="blue")
plt.plot(dates, feels_like, label="Feels Like (°C)", marker="s", linestyle="--", color="orange")
plt.plot(dates, temp_min, label="Min Temperature (°C)", marker="^", linestyle=":", color="green")
plt.plot(dates, temp_max, label="Max Temperature (°C)", marker="v", linestyle="-.", color="red")
plt.title("Weather Forecast", fontsize=16, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(alpha=0.4, linestyle="--")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Extract predictions data
prediction_dates = [item["date"] for item in data["predictions"]]
flavors = list(data["predictions"][0].keys())[1:]  # Exclude the date
flavor_data = {flavor: [item[flavor] for item in data["predictions"]] for flavor in flavors}

# Plot Predictions
plt.figure(figsize=(12, 8))
for flavor, values in flavor_data.items():
    plt.plot(prediction_dates, values, label=flavor, marker="o", linestyle="-")

plt.title("Predictions for Ice Cream Flavors", fontsize=16, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales Predictions", fontsize=12)
plt.grid(alpha=0.4, linestyle="--")
plt.legend(title="Ice Cream Flavors", fontsize=10)
plt.tight_layout()
plt.show()