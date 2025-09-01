from dotenv import load_dotenv
import os
import requests
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("WEATHERAPI_KEY")
BASE_URL = "http://api.weatherapi.com/v1"
DEFAULT_CITY = os.getenv("DEFAULT_CITY") 

def fetch_current_weather(city):
    if not API_KEY:
        return "Weather API key is not set."
    
    if city == "default":
        if not DEFAULT_CITY:
            return "Default city is not set in environment variables."
        city = DEFAULT_CITY

    url = f"{BASE_URL}/current.json"
    params = {
        "key": API_KEY,
        "q": city,
        "aqi": "no"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        location = data['location']['name']
        region = data['location']['region']
        condition = data['current']['condition']['text']
        temp_c = data['current']['temp_c']
        temp_f = data['current']['temp_f']
        humidity = data['current']['humidity']

        return (
            f"The current weather in {location}, {region} is {condition}, "
            f"temperature {temp_f}°F ({temp_c}°C), humidity {humidity}%."
        )
    except Exception as e:
        return f"Sorry, I couldn't get the current weather information. ({e})"

def fetch_weather_forecast(city, days=1):
    """
    Returns a multi-line string with daily forecast, now including:
    - sunrise and sunset times
    - moon phase

    Times are those provided by WeatherAPI for the location's local time.
    """
    if not API_KEY:
        return "Weather API key is not set."
    
    if city == "default":
        if not DEFAULT_CITY:
            return "Default city is not set in environment variables."
        city = DEFAULT_CITY

    url = f"{BASE_URL}/forecast.json"
    params = {
        "key": API_KEY,
        "q": city,
        "days": days,
        "aqi": "no",
        "alerts": "no"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        location = data['location']['name']
        region = data['location']['region']
        forecast_days = data['forecast']['forecastday']

        forecast_str = f"Weather forecast for {location}, {region}:\n"

        for i, day in enumerate(forecast_days):
            date_obj = datetime.strptime(day['date'], "%Y-%m-%d").date()
            day_of_week = date_obj.strftime("%A")

            # Relative label
            if i == 0:
                relative_label = " (today)"
            elif i == 1:
                relative_label = " (tomorrow)"
            else:
                relative_label = ""

            condition = day['day']['condition']['text']
            max_temp_f = day['day']['maxtemp_f']
            min_temp_f = day['day']['mintemp_f']
            max_temp_c = day['day']['maxtemp_c']
            min_temp_c = day['day']['mintemp_c']

            # Astro details (sunrise/sunset/moon phase)
            astro = day.get('astro', {})
            sunrise = astro.get('sunrise', 'N/A')
            sunset = astro.get('sunset', 'N/A')
            moon_phase = astro.get('moon_phase', 'N/A')

            forecast_str += (
                f"{day_of_week} {date_obj}{relative_label}: {condition}, "
                f"High {max_temp_f}°F ({max_temp_c}°C), "
                f"Low {min_temp_f}°F ({min_temp_c}°C). "
                f"Sunrise {sunrise}, Sunset {sunset}, Moon: {moon_phase}.\n"
            )

        return forecast_str.strip()

    except Exception as e:
        return f"Sorry, I couldn't get the weather forecast information. ({e})"
