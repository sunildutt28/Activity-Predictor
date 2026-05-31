import streamlit as st
from recommender_ml import recommend_activity_ml
from datetime import datetime

st.set_page_config(page_title="Activity Suggester", page_icon="🎯")
st.title("🎯 Smart Activity Suggester")
st.markdown("Using Machine Learning to recommend activities based on weather and time")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Weather Conditions")
    temperature = st.slider("Temperature (°C)", -10, 45, 18)
    rain = st.slider("Rain (mm)", 0.0, 50.0, 0.0, step=0.5)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 80, 10)

with col2:
    st.subheader("⏰ Time")
    
    # Option to use current time with timezone fix
    use_current = st.checkbox("Use current time (your local time)")
    
    if use_current:
        # Get current UTC time
        utc_now = datetime.utcnow()
        utc_hour = utc_now.hour
        
        # MANUALLY SET YOUR TIMEZONE OFFSET HERE
        # For Ireland/UK (March-October): +1
        # For Ireland/UK (November-February): +0
        # For US Eastern: -5 or -4
        # For India: +5.5
        
        YOUR_OFFSET = 1  # ← CHANGE THIS TO YOUR TIMEZONE OFFSET
        
        local_hour = (utc_hour + YOUR_OFFSET) % 24
        
        st.info(f"🕐 UTC time: {utc_hour}:00 → Your local time: {local_hour}:00")
        hour = float(local_hour)
    else:
        hour = st.slider("Hour of Day", 0, 23, 12)

# Get recommendation button
if st.button("🎯 Get Activity Suggestion", type="primary", use_container_width=True):
    with st.spinner("🤖 Consulting ML model..."):
        weather_data = {
            "temperature": temperature,
            "rain": rain,
            "windspeed": wind_speed,
            "hour": hour
        }
        
        # Show debug info (remove after testing)
        st.caption(f"🔍 Model input: Hour={hour:.0f}, Temp={temperature}°C, Rain={rain}mm")
        
        activity, confidence, reason, prediction = recommend_activity_ml(weather_data)
        
        st.success(f"### {activity}")
        st.info(f"📝 **Why?** {reason}")
        st.progress(confidence)
        st.caption(f"🤖 ML Confidence: {confidence:.1%}")