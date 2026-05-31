#Clients to test connction to tools registered and access to MCP Server
# mcp_client.py  — correct async client using stdio transport
"""
the client script automatically 
spawns the server as a subprocess via StdioServerParameters. 
So the flow is:
You run client.py
    └── client spawns server.py as a subprocess (stdin/stdout pipe)
        └── handshake happens
            └── tool call goes through
                └── result comes back
                    └── both processes exit
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = r"E:\\Lesson_2_demos\\MCP\\mcsp_setup.py"

async def main():
    # Tell the client to launch your server as a subprocess via stdin/stdout
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Handshake
            await session.initialize()

            # Optional: list available tools to confirm registration
            tools = await session.list_tools()
            print("Registered tools:", [t.name for t in tools.tools])

            # Call your tool
            result = await session.call_tool(
                "google_search",
                arguments={"query": "About Dubai"}
            )

            print("\nResult:")
            for block in result.content:
                print(block.text)

if __name__ == "__main__":
    asyncio.run(main())
