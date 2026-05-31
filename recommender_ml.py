# recommender_ml.py - Complete training with all value combinations
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_comprehensive_training_data():
    """Generate comprehensive training data with all value combinations"""
    print("🔄 Generating comprehensive training data...")
    
    X = []
    y = []
    
    # Define all possible values for each feature
    temperatures = [0, 5, 10, 12, 15, 18, 20, 22, 25, 28, 30, 35]
    rain_values = [0, 1, 2, 5, 8, 10, 15, 20, 30, 50]
    wind_values = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    day_hours = list(range(8, 20))  # 8-19
    night_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    
    # ===== DAYTIME - OUTDOOR (0) =====
    # Good weather conditions: low rain, low wind, comfortable temp
    print("  Adding Outdoor samples...")
    for hour in day_hours:
        for temp in [15, 18, 20, 22, 25]:
            for rain in [0, 1, 2]:
                for wind in [0, 5, 10]:
                    X.append([temp, rain, wind, hour])
                    y.append(0)
        
        # Also add moderate temperatures
        for temp in [12, 14, 16, 17]:
            for rain in [0, 1]:
                for wind in [5, 10]:
                    X.append([temp, rain, wind, hour])
                    y.append(0)
    
    # ===== DAYTIME - INDOOR (1) =====
    # Bad weather: high rain, high wind, or very cold
    print("  Adding Indoor samples...")
    for hour in day_hours:
        # Heavy rain scenarios
        for rain in [10, 15, 20, 30, 50]:
            for temp in [10, 15, 20]:
                for wind in [10, 15, 20]:
                    X.append([temp, rain, wind, hour])
                    y.append(1)
        
        # Moderate rain
        for rain in [5, 8]:
            for temp in [10, 15]:
                for wind in [15, 20]:
                    X.append([temp, rain, wind, hour])
                    y.append(1)
        
        # High wind scenarios
        for wind in [25, 30, 40, 50]:
            for temp in [15, 18, 20]:
                for rain in [0, 2, 5]:
                    X.append([temp, rain, wind, hour])
                    y.append(1)
        
        # Cold weather scenarios
        for temp in [0, 2, 4, 6, 8]:
            for rain in [0, 2, 5]:
                for wind in [5, 10, 15]:
                    X.append([temp, rain, wind, hour])
                    y.append(1)
        
        # Light rain but cold or windy
        for rain in [3, 4, 6]:
            for temp in [8, 10, 12]:
                for wind in [15, 20]:
                    X.append([temp, rain, wind, hour])
                    y.append(1)
    
    # ===== NIGHT (2) =====
    # All night hours are night regardless of weather
    print("  Adding Night samples...")
    for hour in night_hours:
        for temp in [0, 5, 8, 10, 12, 15, 18, 20]:
            for rain in [0, 2, 5, 10, 15, 20]:
                for wind in [5, 10, 15, 20]:
                    X.append([temp, rain, wind, hour])
                    y.append(2)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Generated {len(X)} training samples")
    print(f"  Outdoor (0): {sum(y==0)} samples")
    print(f"  Indoor (1): {sum(y==1)} samples")
    print(f"  Night (2): {sum(y==2)} samples")
    
    # Export to CSV for review
    df = pd.DataFrame(X, columns=["temperature", "rain", "wind_speed", "hour"])
    df["activity_label"] = y
    df["activity_name"] = df["activity_label"].map({0: "Outdoor", 1: "Indoor", 2: "Night"})
    df.to_csv("training_data_comprehensive.csv", index=False)
    print(f"\n✅ Comprehensive training data exported to 'training_data_comprehensive.csv'")
    
    # Show some specific samples
    print("\n🔍 Sample data for moderate values (15°C, 15mm rain, 10km/h wind):")
    moderate_samples = df[(df['temperature'] == 15) & (df['rain'] == 15) & (df['wind_speed'] == 10)]
    if len(moderate_samples) > 0:
        print(moderate_samples[['hour', 'temperature', 'rain', 'wind_speed', 'activity_name']].head(10))
    else:
        print("  No samples found with these exact values")
        print("  Checking for nearby values...")
        nearby = df[(df['temperature'].between(14, 16)) & (df['rain'].between(14, 16)) & (df['wind_speed'].between(9, 11))]
        print(nearby[['hour', 'temperature', 'rain', 'wind_speed', 'activity_name']].head(10))
    
    return X, y

def train_model():
    """Train model using comprehensive training data"""
    print("="*60)
    print("TRAINING MODEL WITH COMPREHENSIVE DATA")
    print("="*60)
    
    # Generate comprehensive training data
    X, y = generate_comprehensive_training_data()
    
    # Shuffle the data
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("\n🔄 Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8
    )
    model.fit(X_scaled, y)
    
    # Save model and scaler
    joblib.dump(model, "weather_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    with open("model_type.txt", "w") as f:
        f.write("GradientBoosting_Comprehensive")
    
    print("\n✅ Model and scaler saved successfully!")
    
    # Test specific scenarios
    print("\n" + "="*60)
    print("TESTING SPECIFIC SCENARIOS")
    print("="*60)
    
    test_scenarios = [
        # (temp, rain, wind, hour, description)
        (15, 0, 10, 10, "15°C, no rain, 10km/h wind at 10am"),
        (15, 5, 10, 10, "15°C, light rain, 10km/h wind at 10am"),
        (15, 10, 10, 10, "15°C, moderate rain, 10km/h wind at 10am"),
        (15, 15, 10, 10, "15°C, heavy rain, 10km/h wind at 10am"),
        (15, 20, 10, 10, "15°C, very heavy rain, 10km/h wind at 10am"),
        (20, 0, 10, 14, "20°C, sunny at 2pm"),
        (20, 15, 10, 14, "20°C, rainy at 2pm"),
        (5, 0, 10, 14, "5°C, cold at 2pm"),
        (15, 0, 10, 22, "15°C at 10pm (night)"),
        (12, 0, 10, 11, "12°C, cool morning"),
        (18, 8, 15, 15, "18°C, moderate rain, 3pm"),
    ]
    
    for temp, rain, wind, hour, desc in test_scenarios:
        test_input = scaler.transform([[temp, rain, wind, hour]])
        pred = model.predict(test_input)[0]
        proba = model.predict_proba(test_input)[0]
        confidence = max(proba)
        pred_name = {0: "OUTDOOR", 1: "INDOOR", 2: "NIGHT"}[pred]
        
        # Determine expected
        if hour >= 20 or hour <= 5:
            expected = "NIGHT"
        elif rain >= 8 or wind >= 25 or temp < 8:
            expected = "INDOOR"
        else:
            expected = "OUTDOOR"
        
        status = "✓" if pred_name == expected else "⚠️"
        print(f"\n{desc}:")
        print(f"  → {pred_name} (conf: {confidence:.1%}) {status}")
        print(f"    Probabilities - O: {proba[0]:.2f}, I: {proba[1]:.2f}, N: {proba[2]:.2f}")
    
    return model, scaler

def load_or_train_model():
    """Load existing model or train if not exists"""
    model_path = "weather_model.pkl"
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print(f"✅ Model loaded from {model_path}")
            
            # Check if it's the comprehensive model
            if os.path.exists("model_type.txt"):
                with open("model_type.txt", "r") as f:
                    model_type = f.read().strip()
                print(f"   Model type: {model_type}")
            
            return model, scaler
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            return train_model()
    else:
        print("⚠️ No existing model found. Training new model...")
        return train_model()

# Load model and scaler at module level
model, scaler = load_or_train_model()

def recommend_activity_ml(weather_data):
    """
    Pure ML recommendation - no rule-based overrides
    """
    try:
        temp = float(weather_data["temperature"])
        rain = float(weather_data["rain"])
        wind = float(weather_data["windspeed"])
        hour = float(weather_data["hour"])
    except (KeyError, ValueError, TypeError) as e:
        print(f"⚠️ Error parsing input: {e}")
        return "📚 Library or Cafe", 0.5, "Unable to process weather data", 1
    
    hour = hour % 24
    
    # Pure ML prediction
    features = np.array([[temp, rain, wind, hour]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    try:
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
    except:
        confidence = 0.85
    
    # Map prediction to activity
    if prediction == 0:  # Outdoor
        if 11 <= hour <= 14:
            if temp > 22:
                activity = "🏖️ Beach Picnic Lunch"
                reason = f"Warm {temp:.1f}°C at {hour:.0f}:00 - perfect for outdoor lunch"
            elif temp > 15:
                activity = "🌳 Park Picnic or Hiking"
                reason = f"Pleasant {temp:.1f}°C at {hour:.0f}:00 - great for midday outdoor activities"
            else:
                activity = "🥾 Light Hiking or Walk"
                reason = f"Cool {temp:.1f}°C at {hour:.0f}:00 - good for active outdoor time"
        elif hour < 11:
            activity = "🚶‍♂️ Morning Walk or Jogging"
            reason = f"Nice morning weather at {hour:.0f}:00 for exercise"
        else:
            activity = "🌲 Hiking or Outdoor Sports"
            reason = f"Good weather at {hour:.0f}:00 for outdoor activities"
    
    elif prediction == 1:  # Indoor
        if rain > 8:
            activity = "🎬 Movie Theater or Museum"
            reason = f"ML detected rain ({rain:.1f}mm) at {hour:.0f}:00 - best indoors"
        elif wind > 25:
            activity = "🛍️ Shopping Mall"
            reason = f"ML detected strong wind ({wind:.1f}km/h) at {hour:.0f}:00 - stay inside"
        elif temp < 8:
            activity = "🏋️‍♂️ Indoor Gym"
            reason = f"ML detected cold ({temp:.1f}°C) at {hour:.0f}:00 - exercise indoors"
        else:
            activity = "📚 Library or Cafe"
            reason = f"ML classified as indoor activity at {hour:.0f}:00"
    
    else:  # Night
        if hour >= 22 or hour <= 4:
            activity = "🛌 Cozy Night In"
            reason = f"It's {hour:.0f}:00 - late night, perfect for resting"
        elif hour >= 20:
            activity = "🍽️ Dinner or Movie"
            reason = f"Evening {hour:.0f}:00 - great for indoor dining or entertainment"
        else:
            activity = "🏠 Home Activities"
            reason = f"Nighttime at {hour:.0f}:00 - enjoy time at home"
    
    return activity, confidence, reason, prediction

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL TRAINING")
    print("="*60)
    model, scaler = train_model()
    
    print("\n" + "="*60)
    print("READY TO RUN STREAMLIT APP")
    print("="*60)
    print("\nRun: streamlit run app.py")