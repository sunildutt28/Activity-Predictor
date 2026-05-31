# recommender_ml.py - COMPLETE VERSION FOR STREAMLIT CLOUD
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def train_model():
    """Train model with CORRECTED daytime activity logic"""
    print("🔄 Training model with proper daytime logic...")
    
    # Features: [temperature, rain_mm, wind_speed, hour_of_day]
    # Labels: 0=Outdoor Day, 1=Indoor Day, 2=Night Activity
    
    X = np.array([
        # ===== OUTDOOR DAY (label 0) - GOOD WEATHER, DAYTIME HOURS 8-18 =====
        # Morning outdoor (8-11)
        [20, 0, 5, 8], [22, 0, 4, 9], [21, 0, 6, 10], [19, 0, 8, 11],
        # Midday outdoor (12-14)
        [25, 0, 3, 12], [27, 0, 4, 12], [24, 0, 5, 13], [26, 0, 3, 13], [28, 0, 4, 14],
        # Afternoon outdoor (15-18)
        [23, 0, 6, 15], [22, 0, 7, 16], [20, 0, 8, 17], [19, 0, 9, 18],
        
        # ===== INDOOR DAY (label 1) - BAD WEATHER, DAYTIME HOURS 8-18 =====
        # Rainy days
        [15, 5, 15, 8], [14, 8, 18, 9], [13, 6, 20, 10], [12, 10, 22, 11],
        [15, 4, 16, 12], [14, 7, 19, 12], [13, 9, 21, 13], [12, 5, 18, 14],
        # Windy/rainy mixed
        [15, 6, 17, 15], [14, 8, 20, 16], [13, 4, 19, 17], [12, 7, 22, 18],
        
        # ===== NIGHT ACTIVITY (label 2) - ANY HOUR 20-5 =====
        # Evening (19-23)
        [18, 0, 8, 19], [17, 0, 10, 20], [16, 2, 12, 21], [15, 3, 15, 22],
        [14, 5, 18, 23],
        # Late night (0-5)
        [13, 0, 10, 0], [12, 0, 12, 1], [11, 2, 15, 2],
        [10, 3, 18, 3], [9, 5, 20, 4], [8, 0, 15, 5],
    ])
    
    # Labels matching the data above
    y = np.array([
        # Outdoor day (13 samples)
        0,0,0,0,  # morning
        0,0,0,0,0,  # midday
        0,0,0,0,  # afternoon
        # Indoor day (12 samples)
        1,1,1,1,1,1,1,1,1,1,1,1,
        # Night activity (11 samples)
        2,2,2,2,2,2,2,2,2,2,2,
    ])
    
    # Train model with fixed random state for reproducibility
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
    
    print("✅ Model trained and saved successfully!")
    return model

def load_or_train_model():
    """Load existing model or train if not exists"""
    model_path = "weather_model.pkl"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("✅ Model loaded from file")
            return model
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("🔄 Retraining...")
            return train_model()
    else:
        print("⚠️ No existing model found. Training new model...")
        return train_model()

# Load model at module level
model = load_or_train_model()

def recommend_activity_ml(weather_data):
    """
    Get activity recommendation based on weather and time
    Expected keys: temperature, rain, windspeed, hour
    """
    # Safely extract and convert values
    try:
        temp = float(weather_data["temperature"])
        rain = float(weather_data["rain"])
        wind = float(weather_data["windspeed"])
        hour = float(weather_data["hour"])
    except (KeyError, ValueError, TypeError) as e:
        print(f"⚠️ Error parsing input: {e}")
        print(f"Received: {weather_data}")
        # Return safe default
        return "📚 Library or Cafe", 0.5, "Unable to process weather data", 1
    
    # Validate hour range
    if hour < 0 or hour > 23:
        print(f"⚠️ Invalid hour: {hour}, defaulting to 12")
        hour = 12
    
    # Prepare features and predict
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
                reason = f"Warm {temp:.1f}°C at {hour:.0f}:00 - perfect for outdoor lunch"
            elif temp > 15:
                activity = "🌳 Park Picnic or Hiking"
                reason = f"Pleasant {temp:.1f}°C at {hour:.0f}:00 - great for midday outdoor activities"
            else:
                activity = "🥾 Light Hiking or Walk"
                reason = f"Cool {temp:.1f}°C at {hour:.0f}:00 - good for active outdoor time"
        elif hour < 11:  # Morning
            activity = "🚶‍♂️ Morning Walk or Jogging"
            reason = f"Nice morning weather at {hour:.0f}:00 for exercise"
        else:  # Afternoon (15-18)
            activity = "🌲 Hiking or Outdoor Sports"
            reason = f"Good afternoon weather at {hour:.0f}:00 for outdoor activities"
    
    elif prediction == 1:  # Indoor day
        if rain > 5:
            activity = "🎬 Movie Theater or Museum"
            reason = f"Heavy rain ({rain:.1f}mm) at {hour:.0f}:00 - best indoors"
        elif wind > 25:
            activity = "🛍️ Shopping Mall"
            reason = f"Very windy ({wind:.1f}km/h) at {hour:.0f}:00 - stay inside"
        elif temp < 8:
            activity = "🏋️‍♂️ Indoor Gym"
            reason = f"Cold {temp:.1f}°C at {hour:.0f}:00 - exercise indoors"
        else:
            activity = "📚 Library or Cafe"
            reason = f"Mixed weather at {hour:.0f}:00 - comfortable indoor options"
    
    else:  # prediction == 2 (Night)
        if hour >= 22 or hour <= 4:
            activity = "🛌 Cozy Night In"
            reason = f"It's {hour:.0f}:00 - late night, perfect for resting"
        elif hour >= 19:
            activity = "🍽️ Dinner or Movie"
            reason = f"Evening {hour:.0f}:00 - great for indoor dining or entertainment"
        else:
            activity = "🏠 Home Activities"
            reason = f"Nighttime at {hour:.0f}:00 - enjoy time at home"
    
    return activity, confidence, reason, prediction

# For local testing
if __name__ == "__main__":
    print("\n🧪 Testing model with various scenarios:\n")
    
    test_cases = [
        {"temperature": 22, "rain": 0, "windspeed": 5, "hour": 12},
        {"temperature": 15, "rain": 8, "windspeed": 15, "hour": 13},
        {"temperature": 18, "rain": 0, "windspeed": 10, "hour": 21},
        {"temperature": 8, "rain": 0, "windspeed": 5, "hour": 10},
        {"temperature": 25, "rain": 0, "windspeed": 3, "hour": 14},
    ]
    
    for weather in test_cases:
        activity, confidence, reason, pred = recommend_activity_ml(weather)
        print(f"Hour {weather['hour']:.0f}, {weather['temperature']}°C: {activity}")
        print(f"  {reason}")
        print(f"  (Confidence: {confidence:.1%}, Label: {pred})\n")