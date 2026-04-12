import os
from serpapi import GoogleSearch
#from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")
SERPER_API_KEY = os.getenv("SERPAPI_API_KEY")

#server = Server(name="my-serpapi-server")
mcp = FastMCP()

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


if __name__ == "__main__":
    #server.run(host="0.0.0.0", port=8000)
    mcp.run()
    #mcp.run(host="0.0.0.0", port=8080)
    #print(google_search("places to visit in paris"))
