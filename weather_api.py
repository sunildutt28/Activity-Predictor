# weather_api.py - CORRECTED VERSION
import requests
import json
from datetime import datetime
from typing import Dict, Optional

def get_weather_data(latitude: float, longitude: float) -> Optional[Dict]:
    """
    Fetch weather data from Open-Meteo API.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    
    Returns:
        Dictionary with temperature, precipitation, precipitation_probability, windspeed, hour
    """
    # CORRECTED URL - using proper field names
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,precipitation,precipitation_probability,wind_speed_10m&hourly=temperature_2m,precipitation,precipitation_probability,wind_speed_10m&timezone=auto"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Get current data
        current = data.get('current', {})
        
        # CORRECT FIELD NAMES - these match the API response
        precipitation = current.get('precipitation', 0.0)  # NOT 'precipitation_amount'
        precipitation_probability = current.get('precipitation_probability', 0.0)
        temperature = current.get('temperature_2m', 0.0)
        wind_speed = current.get('wind_speed_10m', 0.0)
        
        # Get current time
        current_time = current.get('time', datetime.now().isoformat())
        try:
            current_hour = datetime.fromisoformat(current_time.replace('Z', '+00:00')).hour
        except:
            current_hour = datetime.now().hour
        
        weather_data = {
            "temperature": temperature,
            "precipitation": precipitation,  # This is the actual amount in mm
            "precipitation_probability": precipitation_probability,  # This is the chance %
            "windspeed": wind_speed,
            "hour": float(current_hour),
            "latitude": latitude,
            "longitude": longitude
        }
        
        return weather_data
        
    except Exception as e:
        print(f"❌ Error fetching weather: {e}")
        return None

def get_weather_by_city(city: str) -> Optional[Dict]:
    """Get weather data for a city name."""
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    
    try:
        response = requests.get(geocode_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            latitude = data['results'][0]['latitude']
            longitude = data['results'][0]['longitude']
            city_name = data['results'][0]['name']
            country = data['results'][0].get('country', '')
            
            weather = get_weather_data(latitude, longitude)
            if weather:
                weather['city'] = f"{city_name}, {country}"
                return weather
        
        print(f"❌ City '{city}' not found")
        return None
        
    except Exception as e:
        print(f"❌ Error geocoding city: {e}")
        return None

def debug_weather_api(latitude: float, longitude: float):
    """Debug function to see the actual API response."""
    print("=" * 60)
    print(f"DEBUGGING OPEN-METEO API")
    print(f"Coordinates: {latitude}, {longitude}")
    print("=" * 60)
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,precipitation,precipitation_probability,wind_speed_10m&hourly=temperature_2m,precipitation,precipitation_probability,wind_speed_10m&timezone=auto"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print("\n📡 Full API Response (first 2000 chars):")
        print(json.dumps(data, indent=2)[:2000])
        
        print("\n" + "=" * 40)
        print("🔍 Available Current Fields:")
        print("=" * 40)
        
        current = data.get('current', {})
        for key in current.keys():
            print(f"  - {key}: {current[key]}")
        
        print("\n" + "=" * 40)
        print("🔍 Precipitation Analysis")
        print("=" * 40)
        
        precipitation = current.get('precipitation', 'NOT FOUND')
        probability = current.get('precipitation_probability', 'NOT FOUND')
        
        print(f"✅ Precipitation (amount): {precipitation} mm")
        print(f"✅ Precipitation Probability: {probability}%")
        
        if probability > 0 and precipitation == 0:
            print("\n⚠️ NOTE: High probability but 0mm amount means:")
            print("   - Scattered/isolated showers possible")
            print("   - Very light rain (<0.1mm) not measured")
        
        print("\n" + "=" * 60)
        
        return get_weather_data(latitude, longitude)
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        return None

if __name__ == "__main__":
    print("🌤️ Open-Meteo Weather API Tester")
    print("-" * 40)
    
    # Test with Dublin coordinates
    latitude = 53.3498
    longitude = -6.2603
    
    print(f"\n📍 Testing for Dublin (lat: {latitude}, lon: {longitude})")
    print("-" * 40)
    
    weather = get_weather_data(latitude, longitude)
    if weather:
        print("\n✅ Current Weather Data:")
        print(f"  Temperature: {weather['temperature']}°C")
        print(f"  Precipitation Amount: {weather['precipitation']} mm")
        print(f"  Precipitation Probability: {weather['precipitation_probability']}%")
        print(f"  Wind Speed: {weather['windspeed']} km/h")
        print(f"  Hour: {weather['hour']}:00")
    
    # Run debug
    debug_weather_api(latitude, longitude)