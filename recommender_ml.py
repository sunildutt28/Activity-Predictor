# recommender_ml.py - Complete working version
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    """Train the model from scratch"""
    print("🔄 Training model on Streamlit Cloud...")
    
    # Training data (completely self-contained)
    X = np.array([
        # Outdoor day (label 0) - good weather, daytime  
        [25,0,5,10], [28,0,8,11], [30,0,6,12], [22,0,12,13], [26,0,7,14],
        [24,0.2,10,15], [29,0,4,16], [27,0,9,10], [23,0,11,14], [21,0,13,15],
        
        # Indoor day (label 1) - bad weather, daytime
        [15,3,20,10], [12,5,25,11], [10,8,30,12], [8,10,28,13], [14,4,22,14],
        [11,6,26,15], [9,7,29,16], [13,5,24,10], [16,4,18,13], [7,12,32,12],
        
        # Night activities (label 2) - any weather at night
        [25,0,5,22], [22,0,12,23], [18,0.5,15,0], [15,2,20,1], [10,5,25,2],
        [5,8,28,3], [0,10,30,4], [20,0,10,20], [23,0,8,21], [19,1,14,22],
        [14,3,18,23], [8,6,22,0], [3,9,26,1], [12,2,16,20], [28,0,6,21],
        
        # Early morning variations
        [18,0,10,6], [15,0.5,12,7], [12,2,15,8], [10,5,18,9], [14,0,11,6],
        [11,1,13,7], [9,3,16,8], [7,6,19,9], [13,0,12,7], [16,0.5,11,8]
    ])
    
    y = np.array([
        0,0,0,0,0,0,0,0,0,0,  # Outdoor day
        1,1,1,1,1,1,1,1,1,1,  # Indoor day
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  # Night
        0,0,0,1,0,1,1,1,0,0   # Early morning
    ])
    
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "weather_model.pkl")
    print("✅ Model trained and saved!")
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
    temp = weather_data["temperature"]
    rain = weather_data["rain"]
    wind = weather_data["windspeed"]
    hour = weather_data["hour"]
    
    features = np.array([[temp, rain, wind, hour]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    # Activity mapping based on prediction
    if prediction == 0:  # Outdoor day
        if temp > 22 and rain < 0.5 and wind < 10:
            activity = "🏖️ Beach or Swimming Pool"
            reason = "Perfect warm weather for water activities"
        elif temp > 18:
            activity = "🌳 Hiking or Park Picnic"
            reason = "Mild temperature ideal for outdoor exploration"
        elif temp > 12:
            activity = "🚶‍♂️ Jogging or Brisk Walk"
            reason = "Comfortable for exercise"
        else:
            activity = "☕ Outdoor Cafe"
            reason = "Cool but pleasant outdoor conditions"
    
    elif prediction == 1:  # Indoor day
        if rain > 5 or wind > 25:
            activity = "🎬 Movie Theater or Museum"
            reason = "Bad weather outside - perfect for indoor entertainment"
        elif temp < 8:
            activity = "🏋️‍♂️ Indoor Gym or Sports Center"
            reason = "Too cold for outdoor activities"
        else:
            activity = "🛍️ Shopping Mall or Library"
            reason = "Mixed conditions - indoor recommended"
    
    else:  # prediction == 2 (Night)
        if hour >= 22 or hour <= 4:
            activity = "🛌 Cozy Night In"
            reason = f"It's {hour}:00 - late night, perfect for resting"
        else:
            activity = "🎬 Late Night Movie"
            reason = f"Evening time ({hour}:00) - enjoy indoor activities"
    
    return activity, confidence, reason, prediction