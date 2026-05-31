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
    
        # CORRECTED TRAINING DATA with proper hour distribution
    X = np.array([
        # ===== OUTDOOR DAY (label 0) - HOURS 8-18 =====
        # Morning outdoor (8-11)
        [20, 0, 5, 8], [22, 0, 4, 9], [21, 0, 6, 10], [19, 0, 8, 11],
        # Midday outdoor (12-14)
        [25, 0, 3, 12], [27, 0, 4, 12], [24, 0, 5, 13], [26, 0, 3, 13], [28, 0, 4, 14],
        # Afternoon outdoor (15-18)
        [23, 0, 6, 15], [22, 0, 7, 16], [20, 0, 8, 17], [19, 0, 9, 18],
        
        # ===== INDOOR DAY (label 1) - HOURS 8-18 (bad weather) =====
        [15, 5, 15, 8], [14, 8, 18, 9], [13, 6, 20, 10], [12, 10, 22, 11],
        [15, 4, 16, 12], [14, 7, 19, 12], [13, 9, 21, 13], [12, 5, 18, 14],
        [15, 6, 17, 15], [14, 8, 20, 16], [13, 4, 19, 17], [12, 7, 22, 18],
        
        # ===== NIGHT (label 2) - HOURS 19-23 & 0-5 =====
        [18, 0, 8, 19], [17, 0, 10, 20], [16, 2, 12, 21], [15, 3, 15, 22],
        [14, 5, 18, 23], [13, 0, 10, 0], [12, 0, 12, 1], [11, 2, 15, 2],
        [10, 3, 18, 3], [9, 5, 20, 4], [8, 0, 15, 5],
    ])

    y = np.array([
        # Outdoor: 13 samples (hours 8-18)
        0,0,0,0,  # morning
        0,0,0,0,0,  # midday
        0,0,0,0,  # afternoon
        # Indoor: 12 samples (hours 8-18)
        1,1,1,1,1,1,1,1,1,1,1,1,
        # Night: 11 samples (hours 19-5)
        2,2,2,2,2,2,2,2,2,2,2,
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