import os
import re
import requests
from serpapi import GoogleSearch
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")
SERPER_API_KEY = os.getenv("SERPAPI_API_KEY")

#to run isolated instances per client
mcp = FastMCP()
#when running a common server for multiple clients
#mcp = FastMCP(host="0.0.0.0", port=8080)

#@Server.tool()
@mcp.tool()
def google_search(query: str) -> str:
    """
    Search Google and return top 3 results.
    """

    params = {
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
        "engine": "google",
        "num": 3
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    output = []

    for result in results.get("organic_results", [])[:3]:
        title = result.get("title")
        link = result.get("link")
        snippet = result.get("snippet")
        output.append(f"{title}\n{link}\n{snippet}\n")

    return "\n".join(output)

@mcp.tool()
def calculate(operation: str) -> str:
    """
    Perform mathematical operations like addition, subtraction,
    multiplication, and division.
    Takes a mathematical expression as input e.g. '150+25' or '300/5*2'.
    """
    try:
        # Whitelist: only allow numbers and safe math operators
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', operation):
            return "Error: Only basic math operators allowed (+, -, *, /)"
        result = eval(operation)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid syntax in mathematical expression"


@mcp.tool()
def word_counter(text: str) -> str:
    """
    Count the number of words, characters, and sentences in a given text.
    """
    words = len(text.split())
    characters = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    return f"Words: {words}\nCharacters: {characters}\nSentences: {sentences}"

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Fetch current weather for a given city.
    Returns temperature in Celsius and weather description.
    Example: get_weather("London")
    """
    api_key = os.getenv("WEATHER_API_KEY")  # pull from .env, not hardcoded
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        description = data['weather'][0]['description']
        return f"The current temperature in {city} is {temperature}°C with {description}."
    elif response.status_code == 404:
        return f"Error: City '{city}' not found."
    else:
        return f"Error: Unable to fetch weather data (status {response.status_code})."

#mcp.run() with no arguments defaults to stdio transport, which is what the client expects. 
#It's deliberately "silent" — it just listens on stdin/stdout for a client to connect.
if __name__ == "__main__":
    mcp.run()
    #to run a common server
    #mcp.run(transport="sse")

