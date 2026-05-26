import streamlit as st
from weather_api import get_weather, CITY_TIMEZONES  # Import both
from recommender_ml import recommend_activity_ml
from datetime import datetime
from zoneinfo import ZoneInfo

# City coordinates
cities = {
    "Dublin": (53.3498, -6.2603),
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Tokyo": (35.6895, 139.6917),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Sydney": (-33.8688, 151.2093),
    "Los Angeles": (34.0522, -118.2437),
    "Singapore": (1.3521, 103.8198),
    "Mumbai": (19.0760, 72.8777),
}

st.set_page_config(page_title="ML Weather Activity Suggester", layout="wide")

st.title("🤖 ML-Powered Weather Activity Suggester")
st.markdown("*Automatically detects local time for any city*")

# Sidebar - City selection
city = st.sidebar.selectbox("📍 Select your city", list(cities.keys()))

# Show current time for selected city (dynamic preview)
if city in CITY_TIMEZONES:
    city_tz = ZoneInfo(CITY_TIMEZONES[city])
    preview_time = datetime.now(city_tz)
    st.sidebar.info(f"🕐 Local time in {city}: {preview_time.strftime('%H:%M:%S')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 How it works")
st.sidebar.markdown("""
- Model uses: **Temperature + Rain + Wind + Local Hour**
- Automatically detects timezone for each city
- Trained on weather patterns + time of day
""")

# Main content
if st.button("🌤️ Get Live Recommendation", type="primary", use_container_width=True):
    with st.spinner(f"Fetching live weather data for {city}..."):
        lat, lon = cities[city]
        weather = get_weather(lat, lon, city)
        
        # Display weather metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ Temperature", f"{weather['temperature']}°C")
        col2.metric("💨 Wind Speed", f"{weather['windspeed']} km/h")
        col3.metric("🌧️ Rain (next hour)", f"{weather['rain']} mm")
        col4.metric("🕐 Local Time", weather['city_time'])
        
        # Display timezone info
        st.caption(f"📍 Timezone: {weather['timezone']} | Local hour: {weather['hour']}:00")
        
        # ML Recommendation
        activity, confidence, reason, prediction = recommend_activity_ml(weather)
        
        st.markdown("---")
        st.subheader("🎯 ML Model Recommendation")
        st.success(f"### {activity}")
        st.progress(confidence)
        st.caption(f"Model confidence: {confidence*100:.1f}%")
        st.info(f"📝 {reason}")
        
        # Show what the model learned
        with st.expander("🔬 Why this recommendation?"):
            st.write(f"**Time in {city}:** {weather['hour']}:00")
            st.write(f"**Weather:** {weather['temperature']}°C, {weather['rain']}mm rain, {weather['windspeed']}km/h wind")
            st.write(f"**Model prediction:** {'Outdoor' if prediction == 0 else 'Indoor' if prediction == 1 else 'Night'} activity")
            st.write("**Model learned that hour of day is a key factor for recommendations**")

st.markdown("---")
st.caption("⚡ Pure ML | Features: Temp + Rain + Wind + Local Hour | Timezone-aware for any city")