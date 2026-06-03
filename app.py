# app.py - Complete with probability fallback
import streamlit as st
from recommender_ml import recommend_activity_ml
from weather_api import get_weather_data, get_weather_by_city
from datetime import datetime
import os

st.set_page_config(page_title="Activity Suggester", page_icon="🎯", layout="wide")

st.title("🎯 Smart Activity Suggester")
st.markdown("Using **Gradient Boosting ML Model** with Open-Meteo API")

# Sidebar
with st.sidebar:
    st.header("📍 Location Settings")
    
    location_method = st.radio("Location input method:", ["🏙️ City Name", "🗺️ Coordinates"], horizontal=True)
    
    if location_method == "🏙️ City Name":
        city = st.text_input("City Name", value="Dublin")
        use_coordinates = False
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input("Latitude", value=53.3498, format="%.4f")
        with col_lon:
            longitude = st.number_input("Longitude", value=-6.2603, format="%.4f")
        use_coordinates = True
    
    st.divider()
    
    if os.path.exists("weather_model.pkl"):
        st.success("✅ ML Model loaded")

# Weather source selection
col1, col2 = st.columns(2)

with col1:
    weather_source = st.radio("Weather source:", ["🌐 Live Weather", "🎮 Manual Input"], horizontal=True)

if weather_source == "🌐 Live Weather":
    with st.spinner("Fetching live weather..."):
        if use_coordinates:
            weather_data = get_weather_data(latitude, longitude)
        else:
            weather_data = get_weather_by_city(city)
        
        if weather_data:
            st.success("✅ Weather data retrieved")
            
            # Get values
            temperature = weather_data['temperature']
            precipitation_amount = weather_data['precipitation']
            precipitation_probability = weather_data['precipitation_probability']
            wind_speed = weather_data['windspeed']
            hour = weather_data['hour']
            
            # Display metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("🌡️ Temperature", f"{temperature:.1f}°C")
            with col_b:
                st.metric("💧 Precipitation", f"{precipitation_amount:.1f} mm")
            with col_c:
                st.metric("🎲 Rain Probability", f"{precipitation_probability:.0f}%")
            with col_d:
                st.metric("💨 Wind Speed", f"{wind_speed:.1f} km/h")
            
            # ===== CRITICAL FIX: Use probability when amount is 0 =====
            effective_precipitation = precipitation_amount
            
            if precipitation_amount == 0 and precipitation_probability >= 30:
                # High chance of scattered showers
                effective_precipitation = 1.0  # Treat as 1mm of rain
                st.warning(f"⚠️ {precipitation_probability:.0f}% chance of rain - treating as light rain (1.0mm)")
            elif precipitation_amount == 0 and precipitation_probability >= 10:
                # Low chance
                effective_precipitation = 0.5
                st.info(f"ℹ️ {precipitation_probability:.0f}% chance of rain - treating as very light rain (0.5mm)")
            elif precipitation_probability >= 70 and precipitation_amount < 1:
                # High probability with very low amount
                effective_precipitation = max(effective_precipitation, 1.0)
                st.caption(f"⚠️ High probability ({precipitation_probability:.0f}%) with low amount - adjusting")
            
            # Show what we're using
            with st.expander("🔍 Rain Processing"):
                st.write(f"**Raw precipitation amount:** {precipitation_amount} mm")
                st.write(f"**Precipitation probability:** {precipitation_probability}%")
                st.write(f"**Effective precipitation for model:** {effective_precipitation} mm")
                
                if precipitation_amount == 0 and precipitation_probability > 0:
                    st.info("💡 Note: 0mm with high probability means scattered/isolated showers")
            
            weather_input = {
                "temperature": temperature,
                "rain": effective_precipitation,  # Using adjusted value
                "windspeed": wind_speed,
                "hour": hour
            }
        else:
            st.error("Failed to fetch weather data")
            st.stop()
else:
    # Manual input
    with col1:
        temperature = st.slider("Temperature (°C)", -10, 45, 18)
        precipitation_value = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, step=0.5)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 10)
    
    with col2:
        time_method = st.radio("Select time method:", ["Manual", "Use current time"], horizontal=True, index=1)
        if time_method == "Use current time":
            current_utc = datetime.utcnow()
            YOUR_TIMEZONE_OFFSET = 1
            local_hour = (current_utc.hour + YOUR_TIMEZONE_OFFSET) % 24
            hour = float(local_hour)
        else:
            hour = st.slider("Hour of Day", 0, 23, 12)
    
    weather_input = {
        "temperature": temperature,
        "rain": precipitation_value,
        "windspeed": wind_speed,
        "hour": hour
    }

# Time of day indicator
if 6 <= hour < 12:
    st.success("🌅 Morning (6AM - 12PM)")
elif 12 <= hour < 17:
    st.success("☀️ Afternoon (12PM - 5PM)")
elif 17 <= hour < 20:
    st.info("🌆 Evening (5PM - 8PM)")
else:
    st.info("🌙 Night (8PM - 6AM)")

# Get recommendation button
if st.button("🎯 Get Activity Suggestion", type="primary", use_container_width=True):
    with st.spinner("🤖 Analyzing with ML model..."):
        
        activity, confidence, reason, prediction = recommend_activity_ml(weather_input)
        
        # Display results
        st.divider()
        
        pred_names = {0: "Outdoor", 1: "Indoor", 2: "Night"}
        pred_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#9C27B0"}
        pred_icons = {0: "🌳", 1: "📚", 2: "🌙"}
        
        st.markdown(f"""
        <div style="background: {pred_colors[prediction]}; padding: 2rem; border-radius: 1rem; text-align: center;">
            <h2 style="color: white;">✨ Recommended Activity ✨</h2>
            <h1 style="color: white; font-size: 2rem;">{pred_icons[prediction]} {activity}</h1>
            <p style="color: white;">Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"📝 {reason}")

st.divider()
st.markdown("🌤️ Weather from Open-Meteo | 0mm with high probability = scattered showers")