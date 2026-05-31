#Clients to test connction to tools registered and access to MCP Server

import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://localhost:8080/sse") as (read, write):
        async with ClientSession(read, write) as session:
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
