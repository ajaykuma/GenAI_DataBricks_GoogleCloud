import requests
import os
from dotenv import load_dotenv

# Load environment variables
from dotenv import load_dotenv

load_dotenv("E:\\Lesson_2_demos\\.env")

# Get SerpAPI key
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not serpapi_api_key:
    raise ValueError("SERPAPI_API_KEY not found in .env file")

def search_google(query):
    url = "https://serpapi.com/search"

    params = {
        "q": query,
        "api_key": serpapi_api_key,
        "engine": "google"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error:", response.text)
        return None

    data = response.json()

    # Extract top organic results
    results = []
    if "organic_results" in data:
        for item in data["organic_results"][:5]:  # top 5 results
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })

    return results


# Example usage
if __name__ == "__main__":
    query = "latest Fashion trends 2026"
    results = search_google(query)

    if results:
        for i, r in enumerate(results, 1):
            print(f"\nResult {i}")
            print("Title:", r["title"])
            print("Link:", r["link"])
            print("Snippet:", r["snippet"])
    else:
        print("No results found.")
