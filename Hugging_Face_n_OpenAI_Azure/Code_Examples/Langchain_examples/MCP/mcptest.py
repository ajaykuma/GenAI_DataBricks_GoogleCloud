#Clients to test connction to tools registered and access to MCP Server
'''
some issue with client
'''
#from mcp.client import Client
#from mcp import Client
from mcp.server.fastmcp import FastMCP

# Start the MCP server as a subprocess
client = Client("python E:\\Lesson_2_demos\\Lesson_4\\Codes\\MCP\\mcsp_setup.py")

# Call tool by name (string), not function
result = client.call_tool("google_search", {"query": "About Paris"})

print(result)
