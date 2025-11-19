

from langchain_google_genai import ChatGoogleGenerativeAI
import os



import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
import json

load_dotenv()

SERVERS = {     
    "excel": {
         "transport": "stdio",
         "command": "uvx",
         "args": ["excel-mcp-server", "stdio"]
      }
}

async def main():

    model = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash',
                google_api_key=os.getenv('GOOGLE_API_KEY'),
                temperature=0,
                max_output_tokens=1000
            )

    # model.invoke('Hello')
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()


    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    print("Available tools:", named_tools.keys())

    llm = model
    llm_with_tools = llm.bind_tools(tools)

    # prompt = "read_data_from_excel 'C:\work_repo\a.xlsx' sheet 'a'"
    prompt = r"read_data_from_excel 'C:\work_repo\a.xlsx' sheet 'a'"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return

    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")

if __name__ == "__main__":
    asyncio.run(main())