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
            print("-" * 40)

            # Call your tool
            print("TEST: Google Search")
            result = await session.call_tool(
                "google_search",
                arguments={"query": "About Paris"}
            )

            print("\nResult:")
            for block in result.content:
                print(block.text)
            print("-" * 40)
            
            # Call calculator
            print("TEST: Calculator")
            result = await session.call_tool("calculate", arguments={"operation": "300/5*2"})
            print("\nResult:")
            for block in result.content:
                print(block.text)
            print("-" * 40)

            # Call word counter
            print("TEST: Word Counter")
            result = await session.call_tool("word_counter", arguments={"text": "Hello world. How are you."})
            print("\nResult:")
            for block in result.content:
                print(block.text)
            print("-" * 40)
            
            #weather check
            print("TEST: Weather")
            result = await session.call_tool("get_weather", arguments={"city": "Berlin"})
            print("\nResult:")
            for block in result.content:
                print(block.text)   
            print("-" * 40)         

if __name__ == "__main__":
    asyncio.run(main())
