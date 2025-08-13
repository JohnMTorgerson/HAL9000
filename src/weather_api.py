from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("WEATHERAPI_KEY")
BASE_URL = "http://api.weatherapi.com/v1"

def fetch_current_weather(city):
    if not API_KEY:
        return "Weather API key is not set."

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
    if not API_KEY:
        return "Weather API key is not set."

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

        today_date = datetime.now().date()
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

            forecast_str += (
                f"{day_of_week} {date_obj}{relative_label}: {condition}, "
                f"High {max_temp_f}°F ({max_temp_c}°C), "
                f"Low {min_temp_f}°F ({min_temp_c}°C).\n"
            )

        return forecast_str.strip()
    except Exception as e:
        return f"Sorry, I couldn't get the weather forecast information. ({e})"
