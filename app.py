# app.py - Complete Streamlit application
import streamlit as st
from recommender_ml import recommend_activity_ml
from datetime import datetime
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Activity Suggester",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Smart Activity Suggester")
st.markdown("Using **Gradient Boosting ML Model** to recommend activities based on weather and time")

# Sidebar for model info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Gradient Boosting Classifier** trained on:
    - Temperature
    - Rainfall
    - Wind speed
    - Time of day
    
    The model predicts:
    - 🌳 Outdoor activities (good weather, daytime)
    - 📚 Indoor activities (bad weather, daytime)
    - 🌙 Night activities (evening/night hours)
    """)
    
    # Show model info
    if os.path.exists("weather_model.pkl"):
        model_size = os.path.getsize("weather_model.pkl")
        st.success(f"✅ ML Model loaded\nSize: {model_size} bytes")
        
        if os.path.exists("model_type.txt"):
            with open("model_type.txt", "r") as f:
                model_type = f.read().strip()
            st.caption(f"Model: {model_type}")
    else:
        st.error("❌ Model file not found")
    
    # Option to view training data
    if os.path.exists("training_data_export.csv"):
        with st.expander("📊 View Training Data"):
            df = pd.read_csv("training_data_export.csv")
            st.write(f"Total samples: {len(df)}")
            st.dataframe(df.head(10))

# Create two columns for inputs
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
    
    # Time input method - "Use current time" is default
    time_method = st.radio(
        "Select time method:",
        ["Manual", "Use current time"],
        horizontal=True,
        index=1
    )
    
    if time_method == "Use current time":
        # Get current time in UTC
        current_utc = datetime.utcnow()
        
        # CHANGE THIS FOR YOUR LOCATION
        YOUR_TIMEZONE_OFFSET = 1  # Ireland/UK is +1 during daylight savings
        
        local_hour = (current_utc.hour + YOUR_TIMEZONE_OFFSET) % 24
        
        st.info(f"🕐 Current local time: {local_hour}:00")
        
        with st.expander("⏰ Timezone Info"):
            st.write(f"UTC time: {current_utc.hour}:00")
            st.write(f"Your local time: {local_hour}:00")
            st.write(f"Timezone offset: {YOUR_TIMEZONE_OFFSET} hours")
            st.caption("Adjust YOUR_TIMEZONE_OFFSET in the code for your location")
        
        hour = float(local_hour)
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
    with st.spinner("🤖 Analyzing with Gradient Boosting ML model..."):
        
        # Prepare weather data
        weather_data = {
            "temperature": temperature,
            "rain": rain,
            "windspeed": wind_speed,
            "hour": hour
        }
        
        # Get recommendation
        activity, confidence, reason, prediction = recommend_activity_ml(weather_data)
        
        # Display results
        st.divider()
        
        # Results card with color based on prediction type
        pred_names = {0: "Outdoor", 1: "Indoor", 2: "Night"}
        pred_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#9C27B0"}
        pred_icons = {0: "🌳", 1: "📚", 2: "🌙"}
        
        st.markdown(f"""
        <div style="
            background: {pred_colors[prediction]};
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <h2 style="color: white; margin: 0;">✨ Recommended Activity ✨</h2>
            <h1 style="color: white; font-size: 2rem; margin: 1rem 0;">{pred_icons[prediction]} {activity}</h1>
            <p style="color: white; font-size: 1.1rem;">📊 Model confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reason
        st.info(f"📝 **Why?** {reason}")
        
        # Show raw prediction for transparency
        with st.expander("🔍 ML Model Details"):
            st.write(f"**Model Prediction:** {pred_names[prediction]} (code: {prediction})")
            st.write(f"**Confidence:** {confidence:.1%}")
            st.write(f"**Input:** {temperature}°C, {rain}mm rain, {wind_speed}km/h wind at {hour:.0f}:00")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    🤖 Powered by Gradient Boosting ML Model | Automatically detects your local time
</div>
""", unsafe_allow_html=True)