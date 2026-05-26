# Activity-Predictor
Activity predictor based on live weather conditions
# 🌤️ ML Weather Activity Suggester

## 🚀 Live Demo

Try the app here: **[activity-predictor-bysunildutt.streamlit.app](https://activity-predictor-bysunildutt.streamlit.app/)**

## 📖 About

This app uses a **pure Machine Learning model** (Random Forest) to recommend activities based on:
- Real-time weather data (temperature, rain, wind)
- Local time of day (automatically detects timezone for any city)

**No hardcoded rules** - the model learned from training data that:
- ✅ 12 PM (noon) with good weather → Hiking/Picnic
- ✅ 11 PM with any weather → Cozy Night In

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Model**: Random Forest (scikit-learn)
- **Weather API**: Open-Meteo (free, no API key)
- **Deployment**: Streamlit Cloud

## 🏙️ Features

- Select from 10+ cities worldwide
- Real-time weather data
- ML-powered activity recommendations
- Timezone-aware (automatically detects local time)
- Confidence scores for predictions

## 📊 Model Details

**Features used:**
- Temperature (°C)
- Rain (mm)
- Wind Speed (km/h)
- Hour of Day (0-23)

**Training data:** Balanced dataset with daytime/nighttime samples

## 🧪 Try It Yourself

1. Visit the [live app](https://activity-predictor-bysunildutt.streamlit.app/)
2. Select your city
3. Click "Get Live Recommendation"
4. See what the ML model suggests!

## 💻 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/activity-predictor
cd activity-predictor
pip install -r requirements.txt
streamlit run app.py