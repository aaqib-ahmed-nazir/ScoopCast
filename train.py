import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

# Load historical weather (hourly) data and sales data.
weather = pd.read_csv('data/temp-history.csv')
sales = sales = pd.read_csv('data/Ajustes de Stock Rio (1).csv',sep=';', decimal=',')


weather.head()


sales.head()


# Remove the trailing " UTC" from the datetime string and parse it
weather['dt_iso'] = pd.to_datetime(
    weather['dt_iso'].str.replace(' UTC', '', regex=False),
    format='%Y-%m-%d %H:%M:%S %z'
)

# Verify the conversion by printing the first few rows
print(weather['dt_iso'].head())


weather.info()


sales.info()


"""
* Checking for missing values: 
"""


# Null values in the sales data
sales.isnull().sum()


# null values in the weather data
weather.isnull().sum()


"""
* Handling missing values in the data: 
    - Will fill columns with a lot of missing values with 0 (assuming the missing means no rain, no sea level measurement, etc.)
"""


weather.fillna(0, inplace=True)


sales.fillna(0, inplace=True)


weather.isnull().sum()


sales.isnull().sum()


# Ensure 'dt_iso' is a datetime object
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], errors='coerce')

# Extract the date from the datetime column
weather['date'] = weather['dt_iso'].dt.date

# Group by date and aggregate the data
daily_weather = weather.groupby('date').agg({
    'temp': 'mean',
    'feels_like': 'mean',
    'temp_min': 'mean',
    'temp_max': 'mean',
    'humidity': 'mean',
    'dew_point': 'mean',
    'wind_speed': 'mean',
    'wind_deg': 'mean',
    'clouds_all': 'mean',
    'visibility': 'mean',  # Use 'mean' for continuous variables like visibility
    'rain_1h': 'sum',      # Use 'sum' for hourly cumulative measures like rain
    'rain_3h': 'sum',
    'snow_1h': 'sum',
    'snow_3h': 'sum'
}).reset_index()


# Preview the aggregated daily weather data
daily_weather.head()


# Preprocess sales data: convert string dates and rename columns if needed.
sales['Ajuste Fec'] = pd.to_datetime(sales['Ajuste Fec'], dayfirst=True)
sales.rename(columns={'Ajuste Fec': 'date'}, inplace=True)
sales['date'] = sales['date'].dt.date


# Merge the two datasets on date.
df = pd.merge(sales, daily_weather, on='date', how='inner')


df.head()


feature_columns = [
    "temp", "feels_like", "temp_min", "temp_max",
    "humidity", "dew_point",
    "wind_speed", "wind_deg",
    "clouds_all", "visibility",
    "rain_1h", "rain_3h", "snow_1h", "snow_3h"
]


# Separate features and target variables.
# Assuming that flavor sales columns start at index 1 and weather features are at the end.
X = df[feature_columns]
y = df.drop(columns=['date', 'temp', 'humidity'])


# Before model training, add non-negative constraints by taking absolute values of targets
y = y.abs()

# Train a multi-output regression model with increased estimators for better accuracy
model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200,  # Increased from 100
    min_samples_leaf=1,
    random_state=42,
    bootstrap=True,
))
model.fit(X, y)

# Remove the old ClippedRegressor class and custom wrapper
# Instead, modify the model's predict method directly
def safe_predict(self, X):
    predictions = self.predict(X)
    return np.maximum(predictions, 0)  # Ensure no negative values

MultiOutputRegressor.safe_predict = safe_predict

# Save the model with the new predict method
joblib.dump(model, 'ice_cream_sales_model.pkl')

# For evaluation, use the safe prediction
y_pred = model.safe_predict(X)
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE with non-negative predictions
rmse = mse**0.5
print(f'Root Mean Squared Error: {rmse}')