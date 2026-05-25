import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # Better for this
import joblib

print("🌤️ Training Balanced ML Model...")

# Create balanced data where hour doesn't perfectly predict the label
data = {
    # Daytime outdoor (label 0) - GOOD WEATHER ONLY
    "temperature": [25, 28, 30, 26, 27, 24, 29, 26, 28, 25],
    "rain_mm": [0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0],
    "wind_speed": [5, 8, 6, 7, 9, 10, 4, 6, 8, 7],
    "hour": [10, 11, 12, 13, 14, 15, 16, 10, 11, 12],
    "activity_label": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    # Daytime indoor (label 1) - BAD WEATHER during DAY
    "temperature2": [15, 12, 10, 8, 14, 11, 9, 13, 10, 12],
    "rain_mm2": [3, 5, 8, 10, 4, 6, 7, 5, 8, 6],
    "wind_speed2": [20, 25, 30, 28, 22, 26, 29, 24, 27, 25],
    "hour2": [10, 11, 12, 13, 14, 15, 16, 10, 11, 12],
    "activity_label2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    
    # Night activities (label 2) - ANY WEATHER at NIGHT
    "temperature3": [25, 22, 18, 15, 10, 5, 0, 20, 23, 19, 14, 8, 3, 12, 28],
    "rain_mm3": [0, 0, 0.5, 2, 5, 8, 10, 0, 0, 1, 3, 6, 9, 2, 0],
    "wind_speed3": [5, 12, 15, 20, 25, 28, 30, 6, 10, 14, 18, 22, 26, 16, 8],
    "hour3": [20, 21, 22, 23, 0, 1, 2, 20, 21, 22, 23, 0, 1, 2, 22],
    "activity_label3": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    
    # **NEW**: Daytime outdoor with BAD weather? NO - that's indoor
    # **NEW**: Night with GOOD weather? Still night (label 2)
    # **NEW**: Early morning (6-9 AM) - special cases
    "temperature4": [18, 15, 12, 10, 14, 16, 13, 11, 9],
    "rain_mm4": [0, 0.5, 2, 5, 0, 0, 1, 3, 4],
    "wind_speed4": [10, 12, 15, 18, 8, 11, 13, 16, 14],
    "hour4": [6, 7, 8, 9, 6, 7, 8, 9, 6],
    "activity_label4": [0, 0, 0, 1, 0, 0, 1, 1, 1],  # Mixed based on weather
}

# Combine data
all_temps = data["temperature"] + data["temperature2"] + data["temperature3"] + data["temperature4"]
all_rains = data["rain_mm"] + data["rain_mm2"] + data["rain_mm3"] + data["rain_mm4"]
all_winds = data["wind_speed"] + data["wind_speed2"] + data["wind_speed3"] + data["wind_speed4"]
all_hours = data["hour"] + data["hour2"] + data["hour3"] + data["hour4"]
all_labels = data["activity_label"] + data["activity_label2"] + data["activity_label3"] + data["activity_label4"]

X = np.array(list(zip(all_temps, all_rains, all_winds, all_hours)))
y = np.array(all_labels)

# Use Random Forest (better for feature importance)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X, y)

# Check feature importance
features = ["Temperature", "Rain", "Wind Speed", "Hour of Day"]
importances = model.feature_importances_

print("\n🔍 Feature Importance (what model learned):")
for feat, imp in zip(features, importances):
    print(f"  - {feat}: {imp*100:.1f}%")

# Save model
joblib.dump(model, "weather_model.pkl")

# Test predictions
print("\n🧪 Test predictions at 11 PM (hour 23):")
test_cases = [
    [25, 0, 5, 23],   # 25°C, no rain - still night activity
    [5, 10, 30, 23],  # 5°C, heavy rain - night activity
    [15, 0, 10, 23],  # 15°C, calm - night activity
]
for temp, rain, wind, hour in test_cases:
    pred = model.predict([[temp, rain, wind, hour]])[0]
    label = {0: "Outdoor Day", 1: "Indoor Day", 2: "Night"}[pred]
    print(f"  Hour {hour}: {temp}°C, rain={rain}mm → {label}")