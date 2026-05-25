import streamlit as st
from weather_api import get_weather
from recommender_ml import recommend_activity_ml
from datetime import datetime

# City coordinates
cities = {
    "Dublin": (53.3498, -6.2603),
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6895, 139.6917),
}

st.set_page_config(page_title="Pure ML Weather Activity Suggester", layout="wide")

st.title("🤖 Pure ML-Powered Weather Activity Suggester")
st.markdown("*Model features: Temperature + Rain + Wind Speed + Hour of Day*")

# Display current time
current_time = datetime.now()
st.info(f"🕐 Current local time: {current_time.strftime('%H:%M:%S')}")

# Sidebar
city = st.sidebar.selectbox("📍 Select your city", list(cities.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤔 How it works")
st.sidebar.markdown("""
1. ML model trained on **4 features**:
   - Temperature
   - Rain amount  
   - Wind speed
   - **Hour of day** ⏰
   
2. Predicts 3 activity types:
   - 🏃 Outdoor day
   - 🏠 Indoor day  
   - 🌙 Night activity
""")

# Main content
if st.button("🌤️ Get ML Recommendation", type="primary", use_container_width=True):
    with st.spinner("Fetching live weather data & running ML prediction..."):
        lat, lon = cities[city]
        weather = get_weather(lat, lon)
        
        # Display weather metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ Temperature", f"{weather['temperature']}°C")
        col2.metric("💨 Wind Speed", f"{weather['windspeed']} km/h")
        col3.metric("🌧️ Rain (next hour)", f"{weather['rain']} mm")
        col4.metric("⏰ Hour", f"{weather['hour']}:00")
        
        # ML Recommendation
        activity, confidence, reason, prediction = recommend_activity_ml(weather)
        
        st.markdown("---")
        
        # Display model's decision
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("🎯 ML Model Recommendation")
            st.success(f"### {activity}")
            st.progress(confidence)
            st.caption(f"Model confidence: {confidence*100:.1f}%")
            st.info(f"📝 {reason}")
        
        with col_right:
            st.markdown("### 🧠 Model Decision Path")
            if prediction == 0:
                st.markdown("**Classified as:** `Outdoor Day`")
                st.caption("Temperature + Hour → Outdoor suitable")
            elif prediction == 1:
                st.markdown("**Classified as:** `Indoor Day`") 
                st.caption("Rain/Wind → Indoor recommended")
            else:
                st.markdown("**Classified as:** `Night Activity`")
                st.caption("Hour feature → Night override")
        
        # Show feature values used
        with st.expander("🔬 See ML Model Input Features"):
            st.write(f"- **Temperature:** {weather['temperature']}°C")
            st.write(f"- **Rain:** {weather['rain']} mm")
            st.write(f"- **Wind Speed:** {weather['windspeed']} km/h")
            st.write(f"- **Hour of Day:** {weather['hour']}:00")
            st.markdown("---")
            st.write("**Model prediction based on learned patterns from training data**")

# Footer
st.markdown("---")
st.caption("⚡ Pure ML Solution | No hardcoded rules | Features: Temp + Rain + Wind + Hour | Decision Tree Classifier")

# Auto-refresh note
if st.sidebar.checkbox("Auto-refresh every 5 minutes"):
    st.sidebar.warning("Page will refresh automatically")
    import time
    time.sleep(300)
    st.rerun()