# app.py - Complete Streamlit application with Model Diagnostic
import streamlit as st
from recommender_ml import recommend_activity_ml
from datetime import datetime
import os
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Activity Suggester",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Smart Activity Suggester")
st.markdown("Using **Machine Learning** (Random Forest) to recommend activities based on weather and time")

# ============ MODEL DIAGNOSTIC SECTION ============
with st.expander("🔬 Model Diagnostic Tool", expanded=False):
    st.subheader("ML Model Health Check")
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        st.markdown("**📁 Model File Status**")
        if os.path.exists("weather_model.pkl"):
            model_size = os.path.getsize("weather_model.pkl")
            model_time = os.path.getmtime("weather_model.pkl")
            from datetime import datetime as dt
            model_date = dt.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"✅ Model file found")
            st.write(f"- Size: {model_size} bytes")
            st.write(f"- Modified: {model_date}")
        else:
            st.error("❌ Model file NOT found!")
    
    with col_diag2:
        st.markdown("**🤖 Model Loading Test**")
        try:
            test_model = joblib.load("weather_model.pkl")
            st.success("✅ Model loaded successfully")
            st.write(f"- Type: {type(test_model).__name__}")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
    
    st.markdown("---")
    st.markdown("**🧪 Direct Model Prediction Test**")
    
    # Test different hours
    test_hours = [8, 11, 13, 15, 17, 19, 21, 23, 1]
    test_temp = st.slider("Test Temperature (°C)", -10, 45, 18, key="test_temp")
    test_rain = st.slider("Test Rain (mm)", 0.0, 50.0, 0.0, key="test_rain")
    test_wind = st.slider("Test Wind (km/h)", 0, 100, 10, key="test_wind")
    
    if st.button("Run Model Diagnostic Test", key="diag_btn"):
        st.markdown("**Results:**")
        
        # Load model
        if os.path.exists("weather_model.pkl"):
            model = joblib.load("weather_model.pkl")
            
            results = []
            for hour in test_hours:
                test_input = np.array([[test_temp, test_rain, test_wind, hour]])
                pred = model.predict(test_input)[0]
                proba = model.predict_proba(test_input)[0]
                confidence = max(proba)
                
                pred_names = {0: "🌳 Outdoor", 1: "📚 Indoor", 2: "🌙 NIGHT"}
                results.append({
                    "Hour": f"{hour}:00",
                    "Prediction": pred_names[pred],
                    "Confidence": f"{confidence:.1%}",
                    "Label": pred
                })
            
            # Display results in a table
            st.table(results)
            
            # Highlight issues
            st.markdown("**⚠️ Issue Check:**")
            issues = []
            for hour in [8, 11, 13, 15, 17]:
                test_input = np.array([[test_temp, test_rain, test_wind, hour]])
                pred = model.predict(test_input)[0]
                if pred == 2:
                    issues.append(f"❌ Hour {hour}:00 predicted as NIGHT (should be day)")
            
            for hour in [21, 23, 1]:
                test_input = np.array([[test_temp, test_rain, test_wind, hour]])
                pred = model.predict(test_input)[0]
                if pred in [0, 1]:
                    issues.append(f"❌ Hour {hour}:00 predicted as DAY (should be night)")
            
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ No obvious issues detected! Model classifies day/night correctly.")
        else:
            st.error("Cannot run diagnostic - model file not found")
    
    st.markdown("---")
    st.markdown("**💡 Tip:** If you see 'Hour 11:00 predicted as NIGHT', your model file is corrupted or old.")

# Sidebar for app info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Random Forest Classifier** trained on:
    - Temperature
    - Rainfall
    - Wind speed
    - Time of day
    
    The model predicts whether conditions are best for:
    - 🌳 Outdoor activities
    - 📚 Indoor activities  
    - 🌙 Night activities
    """)
    
    # Quick model status
    if os.path.exists("weather_model.pkl"):
        model_size = os.path.getsize("weather_model.pkl")
        st.success(f"✅ ML Model loaded\nSize: {model_size} bytes")
    else:
        st.error("❌ Model file not found")

# Main app - two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Weather Conditions")
    
    temperature = st.slider(
        "Temperature (°C)",
        min_value=-10,
        max_value=45,
        value=18,
        help="Current temperature in Celsius"
    )
    
    rain = st.slider(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        help="Amount of rain in millimeters"
    )
    
    wind_speed = st.slider(
        "Wind Speed (km/h)",
        min_value=0,
        max_value=100,
        value=10,
        help="Current wind speed"
    )

with col2:
    st.subheader("⏰ Time Settings")
    
    # Time input method
    time_method = st.radio(
        "Select time method:",
        ["Manual", "Use current time"],
        horizontal=True
    )
    
    if time_method == "Use current time":
        # Get current time in UTC
        current_utc = datetime.utcnow()
        
        # CHANGE THIS FOR YOUR LOCATION
        # Examples:
        # Ireland/UK (March-Oct): +1, (Nov-Feb): 0
        # US Eastern: -5 or -4
        # India: +5.5
        # Australia: +10 or +11
        
        YOUR_TIMEZONE_OFFSET = 1  # Change this to match your timezone
        
        local_hour = (current_utc.hour + YOUR_TIMEZONE_OFFSET) % 24
        
        st.info(f"🕐 UTC time: {current_utc.hour}:00")
        st.info(f"📍 Local time: {local_hour}:00")
        
        hour = float(local_hour)
        
        # Warning if offset is 0
        if YOUR_TIMEZONE_OFFSET == 0:
            st.caption("⚠️ Timezone offset = 0 (UTC)")
    else:
        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12,
            help="Select the hour for the activity"
        )
    
    # Time of day indicator
    if 6 <= hour < 12:
        st.success("🌅 Morning (6AM - 12PM)")
    elif 12 <= hour < 17:
        st.success("☀️ Afternoon (12PM - 5PM)")
    elif 17 <= hour < 20:
        st.info("🌆 Evening (5PM - 8PM)")
    else:
        st.info("🌙 Night (8PM - 6AM)")

# Main action button
if st.button("🎯 Get Activity Suggestion", type="primary", use_container_width=True):
    with st.spinner("🤖 Analyzing weather and time with ML model..."):
        
        # Prepare weather data
        weather_data = {
            "temperature": temperature,
            "rain": rain,
            "windspeed": wind_speed,
            "hour": hour
        }
        
        # Show input debug
        with st.expander("🔍 Input Debug Info"):
            st.write("**Input to ML Model:**")
            st.json(weather_data)
            st.write(f"**Is daytime?** {'Yes (8-18)' if 8 <= hour <= 18 else 'No (night hours)'}")
        
        # Get recommendation
        activity, confidence, reason, prediction = recommend_activity_ml(weather_data)
        
        # Display results
        st.divider()
        
        # Results card
        pred_names = {0: "Outdoor", 1: "Indoor", 2: "Night"}
        pred_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#9C27B0"}
        
        st.markdown(f"""
        <div style="
            background: {pred_colors[prediction]};
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <h2 style="color: white; margin: 0;">✨ Recommended Activity ✨</h2>
            <h1 style="color: white; font-size: 2.5rem; margin: 1rem 0;">{activity}</h1>
            <p style="color: white; font-size: 1.2rem;">📊 Model confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reason
        st.info(f"📝 **Why?** {reason}")
        
        # Show raw prediction
        st.caption(f"🔍 Model raw output: {prediction} = {pred_names[prediction]} activity")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    🤖 Powered by Random Forest Classifier | 
    <a href="#" target="_blank">Model Diagnostic available above</a>
</div>
""", unsafe_allow_html=True)