import pandas as pd
import numpy as np

# Load dataset
file_path = "openweather_weather_airpollution_top3cities_per_country.csv"
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

print("\n--- Basic Information ---")
print(f"Dataset Shape: {df.shape}")
print(f"Number of Rows: {len(df)}")
print(f"Number of Columns: {len(df.columns)}")

# Check target columns
target_cols = [col for col in df.columns if 'pm2_5' in col.lower() or 'aqi' in col.lower() or 'pm25' in col.lower()]
print("\n--- Target and Pollution Related Columns ---")
print(target_cols)

# Check a sample of columns
print("\n--- First 5 Rows Sample ---")
print(df[['country_name', 'city_name', 'temp', 'humidity', 'wind_speed', 'pm2_5', 'aqi']].head())

# Check missing values in key columns
key_cols = ['temp', 'feels_like', 'pressure', 'humidity', 'wind_speed', 'clouds_all', 'visibility', 'city_population', 'pm2_5']
print("\n--- Missing Values in Key Columns ---")
print(df[key_cols].isna().sum())

# Describe key numeric columns
print("\n--- Summary Statistics of Key Columns ---")
print(df[key_cols].describe())

# Check correlation with PM2.5
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()['pm2_5'].sort_values(ascending=False)
print("\n--- Top Correlations with pm2_5 ---")
print(correlation.head(15))
print(correlation.tail(15))
