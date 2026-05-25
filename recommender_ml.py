import joblib
import numpy as np

# Load trained model
model = joblib.load("weather_model.pkl")

# Activity mapping based on model predictions
# 0 = Outdoor day, 1 = Indoor day, 2 = Night activity
def get_activity_description(label, temp, rain, wind, hour):
    if label == 0:  # Outdoor day
        if temp > 22 and rain < 0.5 and wind < 10:
            return "🏖️ Beach or Swimming Pool", "Perfect warm day weather"
        elif temp > 18:
            return "🌳 Hiking or Park Picnic", "Great weather for outdoor exploration"
        elif temp > 12:
            return "🚶‍♂️ Jogging or Brisk Walk", "Comfortable temperature for exercise"
        else:
            return "☕ Outdoor Cafe", "Cool but pleasant outdoor conditions"
    
    elif label == 1:  # Indoor day
        if rain > 5 or wind > 25:
            return "🎬 Movie Theater or Museum", "Bad weather - perfect for indoor entertainment"
        elif temp < 8:
            return "🏋️‍♂️ Indoor Gym or Sports Center", "Too cold outdoors - stay warm inside"
        else:
            return "🛍️ Shopping Mall or Library", "Mixed conditions - indoor activities recommended"
    
    else:  # Night activity (label 2)
        if hour >= 20 or hour <= 4:
            if temp < 10:
                return "🛌 Cozy Night In", f"{hour}:00 - late night in Dublin, perfect for rest"
            else:
                return "🎬 Late Night Movie", f"{hour}:00 - enjoy a relaxing evening indoors"
        elif hour >= 5 and hour <= 7:
            return "☕ Early Morning Coffee Shop", f"{hour}:00 - start your day indoors"
        else:
            return "🏠 Home Activity (Yoga/Reading)", f"{hour}:00 - evening time for home activities"

def recommend_activity_ml(weather_data):
    temp = weather_data["temperature"]
    rain = weather_data["rain"]
    wind = weather_data["windspeed"]
    hour = weather_data["hour"]
    
    # Prepare features for prediction
    features = np.array([[temp, rain, wind, hour]])
    
    # Get prediction and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    # Get activity description based on prediction
    activity, reason = get_activity_description(prediction, temp, rain, wind, hour)
    
    # Add time context to reason
    if prediction == 2:
        time_context = f"It's {hour}:00 - nighttime recommendation"
    elif hour < 12:
        time_context = f"Morning ({hour}:00) - day activity"
    else:
        time_context = f"Afternoon/Evening ({hour}:00) - day activity"
    
    full_reason = f"{time_context}. {reason}"
    
    return activity, confidence, full_reason, prediction