import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# Timezone mapping for cities - MAKE SURE THIS IS AT THE TOP LEVEL (no indentation)
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
    # Get timezone for the selected city
    timezone_str = CITY_TIMEZONES.get(city_name, "Europe/London")
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "current_weather": True,
        "hourly": "temperature_2m,rain,wind_speed_10m",
        "timezone": timezone_str,
        "forecast_days": 1
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    current = data["current_weather"]
    next_hour_rain = data["hourly"]["rain"][0] if data["hourly"]["rain"] else 0
    
    # Get current time in the city's timezone
    city_tz = ZoneInfo(timezone_str)
    city_time = datetime.now(city_tz)
    
    return {
        "temperature": current["temperature"],
        "windspeed": current["windspeed"],
        "weathercode": current["weathercode"],
        "rain": next_hour_rain,
        "hour": city_time.hour,
        "minute": city_time.minute,
        "timezone": timezone_str,
        "city_time": city_time.strftime("%H:%M:%S")
    }