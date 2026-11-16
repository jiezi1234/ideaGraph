from __future__ import annotations

from src.tools.base import ChatIdeaTool
from src.tools.graph.node_neighborhood import create_node_neighborhood_tool


def create_chat_idea_tools(graph_context_provider) -> list[ChatIdeaTool]:
    return [
        create_node_neighborhood_tool(graph_context_provider),
    ]
