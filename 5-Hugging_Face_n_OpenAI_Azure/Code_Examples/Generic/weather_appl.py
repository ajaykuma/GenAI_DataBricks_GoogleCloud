import requests

def get_weather_for_city():
    # 1. Ask user for input
    city = input("Enter city name (and optional country code e.g., 'Paris FR'): ")

    # 2. Your API key (replace with your actual key from OpenWeatherMap)
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"

    # 3. Base URL for current weather data
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # 4. Parameters: city name, units=metric for Celsius, and API key
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # or "imperial" for Fahrenheit
    }

    # 5. Perform the HTTP GET request
    response = requests.get(base_url, params=params)

    # 6. Check if request was successful (status code 200 means OK)
    if response.status_code == 200:
        data = response.json()
        
        # 7. Extract useful weather information
        city_name = data.get("name")
        main_weather = data["weather"][0]["main"]
        description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        # 8. Print nicely formatted weather info
        print(f"\nWeather in {city_name}:")
        print(f"  Condition: {main_weather} — {description}")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        # If city is not found or another error occurred
        print("Error retrieving weather data. Please check city name or API key.")

# Call the function when running the script
if __name__ == "__main__":
    get_weather_for_city()
