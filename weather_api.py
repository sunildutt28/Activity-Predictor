import requests
from datetime import datetime
from zoneinfo import ZoneInfo

CITY_TIMEZONES = {
    "Dublin": "Europe/Dublin",
    "London": "Europe/London",
    "New York": "America/New_York",
    "Tokyo": "Asia/Tokyo",
    "Paris": "Europe/Paris",
    "Berlin": "Europe/Berlin",
    "Sydney": "Australia/Sydney",
    "Los Angeles": "America/Los_Angeles",
    "Singapore": "Asia/Singapore",
    "Mumbai": "Asia/Kolkata",
}

def get_weather(city_lat, city_lon, city_name):
    timezone_str = CITY_TIMEZONES.get(city_name, "Europe/London")
    
    # Correct API endpoint and parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",  # Note: 'current' not 'current_weather'
        "timezone": timezone_str,
        "forecast_days": 1
    }
    
    print(f"Fetching weather for {city_name} at {city_lat}, {city_lon}")  # Debug
    
    response = requests.get(url, params=params)
    data = response.json()
    
    print(f"API Response keys: {data.keys()}")  # Debug - will show in Streamlit logs
    
    # Get current time in the city's timezone
    city_tz = ZoneInfo(timezone_str)
    city_time = datetime.now(city_tz)
    
    # Extract data from the new API format
    current = data.get("current", {})
    
    return {
        "temperature": current.get("temperature_2m", 15.0),
        "windspeed": current.get("wind_speed_10m", 10.0),
        "weathercode": current.get("weather_code", 0),
        "rain": 0,  # Rain data requires different endpoint, keep simple
        "hour": city_time.hour,
        "minute": city_time.minute,
        "timezone": timezone_str,
        "city_time": city_time.strftime("%H:%M:%S")
    }