import requests
import json
import os
from dotenv import load_dotenv

load_dotenv("E:\\Lesson_2_demos\\.env")

weather_api_key = os.getenv("WEATHER_API_KEY")

if not weather_api_key:
    raise ValueError("WEATHER_API_KEY not found in .env file")
    
url = "https://api.openweathermap.org/data/2.5/weather"
params = {
    "q": "berlin",
    "appid": weather_api_key   # Replace with actual key
}

response = requests.get(url, params=params)

# Optional: check for request success
if response.status_code != 200:
    print("Error:", response.text)
    exit()
    
#data = json.loads(response.text)
data = response.json()

# Some APIs may fail without key; guard against missing fields
if "main" in data:
    temp = data["main"]["temp"] - 273.15
    print(f"{temp:.1f}°C")
else:
    print("Weather data not found:", data)


# summary = llm.generate(
#     f"Explain how this weather feels: {temp}"
# )
