"""LangGraph agent that drives Image Label through the MCP server.

Model is set with LABEL_AGENT_MODEL as "<provider>:<model>", e.g.
"anthropic:claude-sonnet-5-5" (default), "openai:gpt-4.1" or "ollama:llama3.1".
The provider's LangChain package must be installed (langchain-anthropic ships with the agent extra).
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"

SYSTEM_PROMPT = """You help users manage images and annotations in the Image Label app.
Coordinates are image pixels with the origin at the top-left; check an image's width and height before annotating.
Before replacing or removing annotations, confirm with the user unless they already asked for exactly that."""


def mcp_server_config() -> dict:
    # Connect to an already running HTTP server (e.g. the one start.sh launches)
    if url := os.environ.get("LABEL_MCP_URL"):
        return {"label-image": {"transport": "streamable_http", "url": url}}
    # Otherwise launch the server with the same interpreter, forwarding API settings
    env = {k: v for k, v in os.environ.items() if k.startswith("LABEL_") or k in ("PATH", "HOME")}
    return {
        "label-image": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "label_image_mcp.server"],
            "env": env,
        }
    }


async def build_graph(client: MultiServerMCPClient):
    tools = await client.get_tools()
    model = init_chat_model(os.environ.get("LABEL_AGENT_MODEL", DEFAULT_MODEL)).bind_tools(tools)

    async def call_model(state: MessagesState) -> dict:
        response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


async def repl() -> None:
    client = MultiServerMCPClient(mcp_server_config())
    app = await build_graph(client)
    config = {"configurable": {"thread_id": "cli"}}
    print("Image Label agent. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        result = await app.ainvoke({"messages": [HumanMessage(user_input)]}, config)
        print(f"\nagent> {result['messages'][-1].text}")


def main() -> None:
    asyncio.run(repl())


if __name__ == "__main__":
    main()
