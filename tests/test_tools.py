def test_node_neighborhood_tool_exports_openai_schema_and_callable():
    from src.tools.graph.node_neighborhood import create_node_neighborhood_tool

    class GraphContextProvider:
        def __init__(self) -> None:
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append((node_id, node_types, relationship_types, page))
            return {
                "entities": [{"id": node_id, "node_type": "Paper"}],
                "relationships": [],
                "page": page,
                "page_size": 20,
                "total_entities": 1,
                "total_relationships": 0,
            }

    provider = GraphContextProvider()
    tool = create_node_neighborhood_tool(provider)

    assert tool.name == "get_node_neighborhood"
    assert "retrieve graph context" in tool.description
    assert "Use when" in tool.description
    assert "selected paper" in tool.description
    assert "query keyword" in tool.description
    assert "chain-of-evidence" in tool.description
    assert "paper -> problem or limitation -> alternative method or paper" in tool.description

    schema = tool.parameters_schema()
    assert schema["required"] == ["node_id"]
    assert "exact Paper.title" in schema["properties"]["node_id"]["description"]
    assert (
        "Allowed values: Paper, Concept, Domain, Problem, Method, Scenario, Innovation"
        in schema["properties"]["node_types"]["description"]
    )
    assert "Neo4j relationship types" in schema["properties"]["relationship_types"]["description"]
    assert "Allowed values:" not in schema["properties"]["relationship_types"]["description"]
    assert schema["properties"]["relationship_types"]["items"]["enum"] == [
        "CONTAINS",
        "CITES",
        "USED_FOR",
        "CONTRIBUTES_TO",
        "DEPENDENT_ON",
        "EVALUATED_WITH",
        "COMPARE_WITH",
        "SUBCLASS_OF",
        "PART_OF",
        "SYNONYM_OF",
    ]
    assert tool.openai_schema()["function"]["name"] == "get_node_neighborhood"

    result = tool.call(
        {
            "node_id": "Graph-guided discovery",
            "node_types": ["Paper"],
            "relationship_types": ["CITES"],
            "page": 2,
        }
    )

    assert provider.calls == [("Graph-guided discovery", ["Paper"], ["CITES"], 2)]
    assert result["entities"][0]["id"] == "Graph-guided discovery"


def test_node_neighborhood_tool_ignores_literal_entity_node_type():
    from src.tools.graph.node_neighborhood import create_node_neighborhood_tool

    class GraphContextProvider:
        def __init__(self) -> None:
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append((node_id, node_types, relationship_types, page))
            return {
                "entities": [{"id": node_id, "node_type": "Paper"}],
                "relationships": [],
                "page": page,
                "page_size": 20,
                "total_entities": 1,
                "total_relationships": 0,
            }

    provider = GraphContextProvider()
    tool = create_node_neighborhood_tool(provider)

    result = tool.call(
        {
            "node_id": "Graph-guided discovery",
            "node_types": ["Paper", "Entity", "Concept"],
            "relationship_types": ["CITES"],
            "page": 1,
        }
    )

    assert provider.calls == [("Graph-guided discovery", ["Paper", "Concept"], ["CITES"], 1)]
    assert result["entities"][0]["id"] == "Graph-guided discovery"


def test_chat_idea_tool_factory_exports_openai_tools():
    from src.tools.chat_idea import create_chat_idea_tools

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            return {
                "entities": [{"id": node_id, "node_type": "Paper"}],
                "relationships": [],
                "page": page,
                "page_size": 20,
                "total_entities": 1,
                "total_relationships": 0,
            }

    tools = create_chat_idea_tools(GraphContextProvider())

    assert [tool.name for tool in tools] == ["get_node_neighborhood"]
    assert [tool.openai_schema()["function"]["name"] for tool in tools] == ["get_node_neighborhood"]
