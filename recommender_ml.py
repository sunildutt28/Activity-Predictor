# recommender_ml.py - Complete working version with fixed midnight issue
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model():
    """Train the model from scratch with balanced night/day data"""
    print("🔄 Training model on Streamlit Cloud...")
    
    # Features: [temperature, rain_mm, wind_speed, hour_of_day]
    # Labels: 0=Outdoor Day, 1=Indoor Day, 2=Night Activity
    
    X = np.array([
        # ===== OUTDOOR DAY (label 0) - ONLY daytime hours 8-18 =====
        [25,0,5,10], [28,0,8,11], [30,0,6,12], [22,0,12,13], [26,0,7,14],
        [24,0.2,10,15], [29,0,4,16], [27,0,9,10], [23,0,11,14], [21,0,13,15],
        [20,0,10,9], [22,0,8,17], [19,0,12,8], [24,0,6,18], [26,0,5,16],
        [25,0,3,11], [27,0,4,12], [28,0,5,13], [26,0,6,14], [24,0,7,15],
        [23,0,8,10], [25,0,6,16], [22,0,9,9], [21,0,10,17], [20,0,11,8],
        
        # ===== INDOOR DAY (label 1) - ONLY daytime hours 8-18 =====
        [15,3,20,10], [12,5,25,11], [10,8,30,12], [8,10,28,13], [14,4,22,14],
        [11,6,26,15], [9,7,29,16], [13,5,24,10], [16,4,18,13], [7,12,32,12],
        [18,2,22,9], [10,15,35,17], [13,8,28,8], [17,5,20,18], [14,10,25,16],
        [12,7,24,11], [9,9,31,12], [11,6,27,13], [8,11,33,14], [10,8,29,15],
        [13,4,21,9], [15,3,19,17], [6,14,36,8], [16,4,22,18], [11,7,26,10],
        
        # ===== NIGHT ACTIVITY (label 2) - hours 20-4 ONLY =====
        # Hour 20 (8 PM)
        [25,0,5,20], [22,0,8,20], [18,0,12,20], [15,2,15,20], [10,5,20,20],
        [28,0,4,20], [23,0,10,20], [19,1,14,20], [14,3,18,20], [8,6,25,20],
        
        # Hour 21 (9 PM)
        [24,0,6,21], [21,0,9,21], [17,0,13,21], [14,2,16,21], [9,5,22,21],
        [27,0,5,21], [22,0,11,21], [18,1,15,21], [13,3,19,21], [7,6,26,21],
        
        # Hour 22 (10 PM)
        [23,0,7,22], [20,0,10,22], [16,0,14,22], [13,2,17,22], [8,5,23,22],
        [26,0,6,22], [21,0,12,22], [17,1,16,22], [12,3,20,22], [6,6,27,22],
        
        # Hour 23 (11 PM)
        [22,0,8,23], [19,0,11,23], [15,0,15,23], [12,2,18,23], [7,5,24,23],
        [25,0,7,23], [20,0,13,23], [16,1,17,23], [11,3,21,23], [5,6,28,23],
        
        # Hour 0 (12 AM / Midnight) - CRITICAL FIX
        [21,0,9,0], [18,0,12,0], [14,0,16,0], [11,2,19,0], [6,5,25,0],
        [24,0,8,0], [19,0,14,0], [15,1,18,0], [10,3,22,0], [4,6,29,0],
        [20,0,10,0], [17,0,13,0], [13,0,17,0], [9,2,20,0], [5,5,26,0],
        
        # Hour 1 (1 AM)
        [20,0,10,1], [17,0,13,1], [13,0,17,1], [10,2,20,1], [5,5,26,1],
        [22,0,9,1], [18,0,14,1], [14,1,18,1], [9,3,23,1], [3,6,30,1],
        
        # Hour 2 (2 AM)
        [19,0,11,2], [16,0,14,2], [12,0,18,2], [9,2,21,2], [4,5,27,2],
        [21,0,10,2], [17,0,15,2], [13,1,19,2], [8,3,24,2], [2,6,31,2],
        
        # Hour 3 (3 AM)
        [18,0,12,3], [15,0,15,3], [11,0,19,3], [8,2,22,3], [3,5,28,3],
        [20,0,11,3], [16,0,16,3], [12,1,20,3], [7,3,25,3], [1,6,32,3],
        
        # Hour 4 (4 AM)
        [17,0,13,4], [14,0,16,4], [10,0,20,4], [7,2,23,4], [2,5,29,4],
        [19,0,12,4], [15,0,17,4], [11,1,21,4], [6,3,26,4], [0,6,33,4],
        
        # ===== EARLY MORNING (hours 5-7) - mixed based on weather =====
        # Hour 5
        [18,0,10,5], [15,0.5,12,5], [12,2,15,5], [10,5,18,5], [14,0,11,5],
        [16,0,9,5], [11,3,16,5], [8,6,20,5], [13,0,13,5], [9,4,19,5],
        
        # Hour 6
        [19,0,9,6], [16,0.5,11,6], [13,2,14,6], [11,5,17,6], [15,0,10,6],
        [17,0,8,6], [12,3,15,6], [9,6,19,6], [14,0,12,6], [10,4,18,6],
        
        # Hour 7
        [20,0,8,7], [17,0.5,10,7], [14,2,13,7], [12,5,16,7], [16,0,9,7],
        [18,0,7,7], [13,3,14,7], [10,6,18,7], [15,0,11,7], [11,4,17,7],
    ])
    
    # Labels matching the data above
    y = np.array([
        # Outdoor day (25 samples)
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        # Indoor day (25 samples)
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        # Night hour 20 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 21 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 22 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 23 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 0 (15 samples) - CRITICAL for midnight
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        # Night hour 1 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 2 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 3 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Night hour 4 (10 samples)
        2,2,2,2,2,2,2,2,2,2,
        # Early morning hour 5 (10 samples - outdoor when good weather)
        0,0,0,1,0,0,1,1,0,1,
        # Early morning hour 6 (10 samples)
        0,0,1,1,0,0,1,1,0,1,
        # Early morning hour 7 (10 samples)
        0,0,1,1,0,0,1,1,0,1,
    ])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "weather_model.pkl")
    
    # Print feature importance for debugging
    features = ["Temperature", "Rain", "Wind Speed", "Hour"]
    importances = model.feature_importances_
    print("📊 Feature importance after training:")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp*100:.1f}%")
    
    print("✅ Model trained and saved successfully!")
    return model

# Load or train model
if os.path.exists("weather_model.pkl"):
    try:
        model = joblib.load("weather_model.pkl")
        print("✅ Model loaded from existing file")
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
    
    # Prepare features for prediction
    features = np.array([[temp, rain, wind, hour]])
    
    # Get prediction and confidence
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    # Map prediction to activity
    if prediction == 0:  # Outdoor day
        if temp > 22 and rain < 0.5 and wind < 10:
            activity = "🏖️ Beach or Swimming Pool"
            reason = f"Perfect weather at {hour}:00 for water activities"
        elif temp > 18:
            activity = "🌳 Hiking or Park Picnic"
            reason = f"Great weather at {hour}:00 for outdoor exploration"
        elif temp > 12:
            activity = "🚶‍♂️ Jogging or Brisk Walk"
            reason = f"Comfortable temperature at {hour}:00 for exercise"
        else:
            activity = "☕ Outdoor Cafe"
            reason = f"Cool but pleasant at {hour}:00 for outdoor seating"
    
    elif prediction == 1:  # Indoor day
        if rain > 5 or wind > 25:
            activity = "🎬 Movie Theater or Museum"
            reason = f"Bad weather at {hour}:00 - perfect for indoor entertainment"
        elif temp < 8:
            activity = "🏋️‍♂️ Indoor Gym or Sports Center"
            reason = f"Too cold at {hour}:00 for outdoor activities"
        else:
            activity = "🛍️ Shopping Mall or Library"
            reason = f"Mixed conditions at {hour}:00 - indoor recommended"
    
    else:  # prediction == 2 (Night activity)
        if hour >= 22 or hour <= 4:
            activity = "🛌 Cozy Night In"
            reason = f"It's {hour}:00 - late night, perfect for resting at home"
        elif hour >= 20:
            activity = "🎬 Late Night Movie"
            reason = f"It's {hour}:00 - evening time for indoor entertainment"
        else:
            activity = "🏠 Home Activity (Yoga/Reading)"
            reason = f"It's {hour}:00 - nighttime, best to stay indoors"
    
    return activity, confidence, reason, prediction


# Debug test - runs when file is executed directly
if __name__ == "__main__":
    print("\n🧪 Testing model at different hours:")
    print("-" * 40)
    test_hours = [0, 1, 2, 3, 4, 5, 6, 7, 12, 14, 20, 21, 22, 23]
    for hour in test_hours:
        pred = model.predict(np.array([[20, 0, 10, hour]]))[0]
        label = {0: "🌳 Outdoor Day", 1: "🏠 Indoor Day", 2: "🌙 Night Activity"}[pred]
        print(f"  Hour {hour:2d}:00 → {label}")
    print("-" * 40)
    print("✅ Midnight (hour 0) should now show Night Activity!")