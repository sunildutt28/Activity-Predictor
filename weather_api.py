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
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        
        # Debug: Print what we received (will appear in Streamlit logs)
        print(f"API Response keys: {data.keys()}")
        
        # Check if 'current_weather' exists
        if "current_weather" not in data:
            error_msg = f"API Error: {data.get('reason', 'Unknown error')}"
            print(error_msg)
            raise KeyError(error_msg)
        
        current = data["current_weather"]
        
        # Safely get hourly rain data
        hourly_rain = data.get("hourly", {}).get("rain", [0])
        next_hour_rain = hourly_rain[0] if hourly_rain else 0
        
        # Get current time in the city's timezone
        city_tz = ZoneInfo(timezone_str)
        city_time = datetime.now(city_tz)
        
        return {
            "temperature": current.get("temperature", 15),
            "windspeed": current.get("windspeed", 10),
            "weathercode": current.get("weathercode", 0),
            "rain": next_hour_rain,
            "hour": city_time.hour,
            "minute": city_time.minute,
            "timezone": timezone_str,
            "city_time": city_time.strftime("%H:%M:%S")
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        # Return fallback data
        city_tz = ZoneInfo(timezone_str)
        city_time = datetime.now(city_tz)
        return {
            "temperature": 15,
            "windspeed": 10,
            "weathercode": 0,
            "rain": 0,
            "hour": city_time.hour,
            "minute": city_time.minute,
            "timezone": timezone_str,
            "city_time": city_time.strftime("%H:%M:%S")
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise