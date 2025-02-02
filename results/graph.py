import matplotlib.pyplot as plt

# Updated data
forecast = [
    {"date": "2025-02-02", "temp": 34.0, "feels_like": 35.0, "temp_min": 22.0, "temp_max": 35.0},
    {"date": "2025-02-03", "temp": 37.0, "feels_like": 38.0, "temp_min": 24.0, "temp_max": 37.0},
    {"date": "2025-02-04", "temp": 34.0, "feels_like": 34.5, "temp_min": 25.0, "temp_max": 34.0},
    {"date": "2025-02-05", "temp": 32.0, "feels_like": 33.0, "temp_min": 23.0, "temp_max": 32.0},
    {"date": "2025-02-06", "temp": 32.0, "feels_like": 33.0, "temp_min": 21.0, "temp_max": 32.0},
    {"date": "2025-02-07", "temp": 37.0, "feels_like": 38.0, "temp_min": 25.0, "temp_max": 37.0},
    {"date": "2025-02-08", "temp": 39.0, "feels_like": 40.0, "temp_min": 26.0, "temp_max": 39.0},
]

predictions = [
    {"date": "2025-02-02", "Americana": 22.05, "Cheesecake de Frambuesa": 17.59, "Chocolate con Almendras": 38.02, "Crema Oreo": 33.40, "Dulce de Leche Granizado": 54.85, "Maracuyá": 22.39},
    {"date": "2025-02-03", "Americana": 21.72, "Cheesecake de Frambuesa": 17.63, "Chocolate con Almendras": 38.02, "Crema Oreo": 33.29, "Dulce de Leche Granizado": 54.45, "Maracuyá": 22.25},
    {"date": "2025-02-04", "Americana": 25.42, "Cheesecake de Frambuesa": 17.61, "Chocolate con Almendras": 36.45, "Crema Oreo": 31.72, "Dulce de Leche Granizado": 48.37, "Maracuyá": 21.62},
    {"date": "2025-02-05", "Americana": 23.36, "Cheesecake de Frambuesa": 18.28, "Chocolate con Almendras": 35.42, "Crema Oreo": 30.57, "Dulce de Leche Granizado": 46.94, "Maracuyá": 19.85},
    {"date": "2025-02-06", "Americana": 22.52, "Cheesecake de Frambuesa": 17.65, "Chocolate con Almendras": 37.85, "Crema Oreo": 33.56, "Dulce de Leche Granizado": 55.94, "Maracuyá": 22.90},
    {"date": "2025-02-07", "Americana": 21.98, "Cheesecake de Frambuesa": 17.48, "Chocolate con Almendras": 37.94, "Crema Oreo": 33.31, "Dulce de Leche Granizado": 54.58, "Maracuyá": 22.25},
    {"date": "2025-02-08", "Americana": 22.04, "Cheesecake de Frambuesa": 17.68, "Chocolate con Almendras": 38.04, "Crema Oreo": 33.31, "Dulce de Leche Granizado": 54.75, "Maracuyá": 22.28},
]

# Weather Forecast Data
dates = [item["date"] for item in forecast]
temps = [item["temp"] for item in forecast]
feels_like = [item["feels_like"] for item in forecast]
temp_min = [item["temp_min"] for item in forecast]
temp_max = [item["temp_max"] for item in forecast]

# Plot Weather Forecast
plt.figure(figsize=(10, 6))
plt.plot(dates, temps, label="Temperature (°C)", marker="o", linestyle="-", color="blue")
plt.plot(dates, feels_like, label="Feels Like (°C)", marker="s", linestyle="--", color="orange")
plt.plot(dates, temp_min, label="Min Temperature (°C)", marker="^", linestyle=":", color="green")
plt.plot(dates, temp_max, label="Max Temperature (°C)", marker="v", linestyle="-.", color="red")
plt.title("Weather Forecast in Roasario, Argentina", fontsize=16, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(alpha=0.4, linestyle="--")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Ice Cream Predictions Data
prediction_dates = [item["date"] for item in predictions]
flavors = list(predictions[0].keys())[1:]
flavor_data = {flavor: [item[flavor] for item in predictions] for flavor in flavors}

# Plot Ice Cream Predictions
plt.figure(figsize=(12, 8))
for flavor, values in flavor_data.items():
    plt.plot(prediction_dates, values, label=flavor, marker="o", linestyle="-")

plt.title("Ice Cream Sales Predictions", fontsize=16, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.grid(alpha=0.4, linestyle="--")
plt.legend(title="Ice Cream Flavors", fontsize=10)
plt.tight_layout()
plt.show()
