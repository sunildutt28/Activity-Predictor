# recommender_ml.py - CORRECTED VERSION
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    """Train model with CORRECT daytime activity logic"""
    print("🔄 Training model with proper daytime logic...")
    
    # Features: [temperature, rain_mm, wind_speed, hour_of_day]
    # Labels: 0=Outdoor Day, 1=Indoor Day, 2=Night Activity
    
    X = np.array([
        # ===== OUTDOOR DAY (label 0) - GOOD WEATHER, DAYTIME HOURS 8-18 =====
        # Hiking weather (12-22°C, low rain)
        [22, 0, 5, 12], [20, 0, 8, 12], [18, 0, 10, 12], [19, 0, 7, 13], [21, 0, 6, 13],
        [23, 0, 4, 14], [22, 0, 5, 14], [20, 0, 8, 15], [19, 0, 9, 15], [18, 0, 10, 11],
        [21, 0, 6, 11], [22, 0, 5, 16], [20, 0, 7, 16], [19, 0, 8, 10], [18, 0, 9, 17],
        
        # Beach weather (warm, >22°C)
        [25, 0, 5, 13], [27, 0, 4, 13], [28, 0, 3, 14], [26, 0, 6, 14], [29, 0, 4, 15],
        [30, 0, 3, 12], [28, 0, 5, 12], [26, 0, 6, 11], [27, 0, 4, 16], [25, 0, 7, 15],
        
        # Cool outdoor (12-17°C, good for walking)
        [16, 0, 8, 14], [15, 0, 10, 14], [14, 0, 12, 15], [17, 0, 7, 15], [15, 0, 9, 11],
        [16, 0, 8, 12], [14, 0, 11, 13], [13, 0, 12, 16], [15, 0, 10, 10], [17, 0, 8, 17],
        
        # ===== INDOOR DAY (label 1) - BAD WEATHER, DAYTIME HOURS 8-18 =====
        # Rainy days
        [15, 3, 15, 12], [14, 5, 18, 13], [13, 8, 20, 14], [12, 10, 22, 12], [14, 4, 16, 15],
        [15, 6, 17, 11], [13, 7, 19, 16], [12, 9, 21, 10], [14, 5, 18, 17], [13, 6, 19, 13],
        
        # Windy days
        [18, 0, 28, 14], [17, 0, 30, 13], [16, 0, 32, 12], [19, 0, 26, 15], [17, 0, 29, 11],
        [18, 0, 27, 16], [16, 0, 31, 10], [19, 0, 25, 17], [17, 0, 28, 14], [18, 0, 29, 12],
        
        # Cold days (<8°C)
        [7, 0, 10, 13], [5, 0, 12, 14], [3, 0, 15, 12], [6, 0, 11, 15], [4, 0, 13, 11],
        [8, 0, 9, 16], [2, 0, 14, 10], [1, 0, 16, 17], [7, 0, 10, 14], [5, 0, 12, 13],
        
        # ===== NIGHT ACTIVITY (label 2) - ANY HOUR 20-5 =====
        # ALL night hours force indoor
        [22, 0, 5, 20], [20, 0, 8, 20], [18, 0, 10, 20], [15, 2, 12, 21], [12, 5, 15, 21],
        [10, 0, 8, 22], [8, 0, 10, 22], [6, 3, 12, 22], [5, 8, 15, 23], [4, 10, 18, 23],
        [25, 0, 5, 0], [22, 0, 8, 0], [20, 0, 10, 0], [18, 0, 12, 1], [15, 2, 15, 1],
        [12, 0, 8, 2], [10, 0, 10, 2], [8, 3, 12, 3], [6, 5, 15, 3], [5, 8, 18, 4],
        [20, 0, 8, 23], [18, 0, 10, 23], [15, 0, 12, 0], [12, 0, 8, 1], [10, 0, 10, 2],
    ])
    
    # Labels matching the data above
    y = np.array([
        # Outdoor day - HIKING weather (15 samples)
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        # Outdoor day - BEACH weather (10 samples)
        0,0,0,0,0,0,0,0,0,0,
        # Outdoor day - COOL weather (10 samples)
        0,0,0,0,0,0,0,0,0,0,
        # Indoor day - RAIN (10 samples)
        1,1,1,1,1,1,1,1,1,1,
        # Indoor day - WINDY (10 samples)
        1,1,1,1,1,1,1,1,1,1,
        # Indoor day - COLD (10 samples)
        1,1,1,1,1,1,1,1,1,1,
        # Night activity (25 samples)
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    ])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "weather_model.pkl")
    
    # Print feature importance
    features = ["Temperature", "Rain", "Wind Speed", "Hour"]
    importances = model.feature_importances_
    print("📊 Feature importance:")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp*100:.1f}%")
    
    print("✅ Model trained successfully!")
    return model

# Load or train model
if os.path.exists("weather_model.pkl"):
    try:
        model = joblib.load("weather_model.pkl")
        print("✅ Model loaded from file")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("🔄 Retraining...")
        model = train_model()
else:
    model = train_model()

def recommend_activity_ml(weather_data):
    """Get activity recommendation based on weather and time"""
    temp = weather_data["temperature"]
    rain = weather_data["rain"]
    wind = weather_data["windspeed"]
    hour = weather_data["hour"]
    
    features = np.array([[temp, rain, wind, hour]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    # Map prediction to activity with time-appropriate suggestions
    if prediction == 0:  # Outdoor day
        # Time-appropriate outdoor suggestions
        if 11 <= hour <= 14:  # Lunch time / midday
            if temp > 22:
                activity = "🏖️ Beach Picnic Lunch"
                reason = f"Warm {temp}°C at {hour}:00 - perfect for outdoor lunch"
            elif temp > 15:
                activity = "🌳 Park Picnic or Hiking"
                reason = f"Pleasant {temp}°C at {hour}:00 - great for midday outdoor activities"
            else:
                activity = "🥾 Light Hiking or Walk"
                reason = f"Cool {temp}°C at {hour}:00 - good for active outdoor time"
        elif hour < 11:  # Morning
            activity = "🚶‍♂️ Morning Walk or Jogging"
            reason = f"Nice morning weather at {hour}:00 for exercise"
        else:  # Afternoon (15-18)
            activity = "🌲 Hiking or Outdoor Sports"
            reason = f"Good afternoon weather at {hour}:00 for outdoor activities"
    
    elif prediction == 1:  # Indoor day
        if rain > 5:
            activity = "🎬 Movie Theater or Museum"
            reason = f"Heavy rain ({rain}mm) at {hour}:00 - best indoors"
        elif wind > 25:
            activity = "🛍️ Shopping Mall"
            reason = f"Very windy ({wind}km/h) at {hour}:00 - stay inside"
        elif temp < 8:
            activity = "🏋️‍♂️ Indoor Gym"
            reason = f"Cold {temp}°C at {hour}:00 - exercise indoors"
        else:
            activity = "📚 Library or Cafe"
            reason = f"Mixed weather at {hour}:00 - comfortable indoor options"
    
    else:  # prediction == 2 (Night)
        if hour >= 22 or hour <= 4:
            activity = "🛌 Cozy Night In"
            reason = f"It's {hour}:00 - late night, perfect for resting"
        elif hour >= 19:
            activity = "🍽️ Dinner or Movie"
            reason = f"Evening {hour}:00 - great for indoor dining or entertainment"
        else:
            activity = "🏠 Home Activities"
            reason = f"Nighttime at {hour}:00 - enjoy time at home"
    
    return activity, confidence, reason, prediction


# Test at 12 PM
if __name__ == "__main__":
    print("\n🧪 Testing model at 12 PM with typical Dublin weather:")
    test_features = np.array([[15, 0, 10, 12]])  # 15°C, no rain, light wind
    pred = model.predict(test_features)[0]
    print(f"  Input: 15°C, 0mm rain, 10km/h wind, 12:00")
    print(f"  Prediction: {'Outdoor' if pred == 0 else 'Indoor' if pred == 1 else 'Night'}")
    print("  Expected: Outdoor (Hiking/Picnic)")