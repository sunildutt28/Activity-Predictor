import requests
from datetime import datetime

def get_weather(city_lat, city_lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "current_weather": True,
        "hourly": "temperature_2m,rain,wind_speed_10m",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    current = data["current_weather"]
    # Get next hour's rain (index 0)
    next_hour_rain = data["hourly"]["rain"][0] if data["hourly"]["rain"] else 0
    
    return {
        "temperature": current["temperature"],
        "windspeed": current["windspeed"],
        "weathercode": current["weathercode"],
        "rain": next_hour_rain,
        "hour": datetime.now().hour  # Add current hour as feature
    }