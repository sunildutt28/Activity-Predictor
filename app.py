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
    # Option to use current time or manual
    use_current = st.checkbox("Use current time")
    if use_current:
        current_hour = datetime.now().hour
        st.info(f"Using current time: {current_hour}:00")
        hour = float(current_hour)
    else:
        hour = st.slider("Hour of Day", 0, 23, 12)

# Add a debug expander
with st.expander("🔧 Debug Info (for troubleshooting)"):
    st.write("Input values being sent to model:")
    st.json({
        "temperature": float(temperature),
        "rain": float(rain),
        "windspeed": float(wind_speed),
        "hour": float(hour)
    })

# Get recommendation button
if st.button("🎯 Get Activity Suggestion", type="primary", use_container_width=True):
    with st.spinner("🤖 Consulting ML model..."):
        # Prepare weather data (ensure correct key names)
        weather_data = {
            "temperature": temperature,
            "rain": rain,
            "windspeed": wind_speed,  # Note: 'windspeed' not 'wind'
            "hour": hour
        }
        
        # Get recommendation
        activity, confidence, reason, prediction = recommend_activity_ml(weather_data)
        
        # Display results
        st.success(f"### {activity}")
        st.info(f"📝 **Why?** {reason}")
        
        # Show confidence as progress bar
        st.progress(confidence)
        st.caption(f"ML Model Confidence: {confidence:.1%}")
        
        # Show prediction label for transparency
        label_map = {0: "Outdoor Day", 1: "Indoor Day", 2: "Night Activity"}
        st.caption(f"🤖 Model classification: {label_map[prediction]}")

# Add footer with ML info
st.divider()
st.caption("Powered by Random Forest Classifier | Trained on weather and time data")