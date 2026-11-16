from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, field_validator

from src.tools.base import ChatIdeaTool


TOOL_NAME = "get_node_neighborhood"
NodeType = Literal["Paper", "Concept", "Domain", "Problem", "Method", "Scenario", "Innovation"]
RelationshipType = Literal[
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
NODE_TYPE_VALUES = get_args(NodeType)
TOOL_DESCRIPTION = (
    "Use when you need to retrieve graph context as graph-grounded evidence "
    "before or during idea generation: "
    "query the selected paper to inspect citations and contained entities, query "
    "a candidate or related paper title to compare nearby context, or query a "
    "query keyword/entity id to explore methods, problems, scenarios, concepts, "
    "domains, or innovations connected to the user's topic. Use it for "
    "chain-of-evidence exploration such as paper -> problem or limitation -> "
    "alternative method or paper -> experiment signal, following useful nodes "
    "from one observation into the next. This tool returns "
    "one-hop neighboring Paper and semantic Entity nodes plus relationships "
    "around one paper title or entity id. If exact and alias lookup miss, the "
    "runtime may fall back to the closest non-Paper entity by vector score and "
    "returns a lookup note that explains the fallback."
)


class NodeNeighborhoodArgs(BaseModel):
    node_id: Annotated[
        str,
        Field(
            description=(
                "The exact Paper.title or Entity.id to query. Prefer the selected "
                "paper title for paper context; use a user keyword only when it is "
                "likely to be an entity id in the graph. If no exact or alias match "
                "exists, the graph runtime may use this text to find the closest "
                "non-Paper entity by vector score."
            ),
        ),
    ]
    node_types: Annotated[
        list[NodeType],
        Field(
            default_factory=list,
            description=(
                "Optional labels to keep among returned one-hop neighbor nodes; the "
                "queried center node itself is not filtered by this option. Leave "
                "empty when any nearby graph node type may be useful. "
                f"Allowed values: {', '.join(NODE_TYPE_VALUES)}."
            ),
        ),
    ]
    relationship_types: Annotated[
        list[RelationshipType],
        Field(
            default_factory=list,
            description=(
                "Optional Neo4j relationship types to keep in the returned "
                "neighborhood. Leave empty when all relationship types may be useful."
            ),
        ),
    ]
    page: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Optional page number for paginated results. Use 1 unless a later page is explicitly needed.",
        ),
    ]
    depth: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Optional search depth. Defaults to 1.",
        ),
    ] = 1

    @field_validator("node_types", mode="before")
    @classmethod
    def _normalize_node_types(cls, value):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            return value
        normalized = []
        seen = set()
        for item in value:
            if item == "Entity":
                continue
            if item not in NODE_TYPE_VALUES:
                raise ValueError(
                    f"node_types must be one of {', '.join(NODE_TYPE_VALUES)}"
                )
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized


def create_node_neighborhood_tool(graph_context_provider) -> ChatIdeaTool:
    def get_node_neighborhood(
        node_id: str,
        node_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        page: int = 1,
        depth: int = 1,
    ) -> dict:
        import inspect
        # 写代码时要加必要的中文注释：如果底层 provider 接收 depth，则传递它以支持多跳查询，否则保持兼容（如 Mock 对象）
        sig = inspect.signature(graph_context_provider.get_node_neighborhood)
        kwargs = {}
        if "depth" in sig.parameters:
            kwargs["depth"] = depth
        return graph_context_provider.get_node_neighborhood(
            node_id=node_id,
            node_types=node_types or [],
            relationship_types=relationship_types or [],
            page=page,
            **kwargs
        )

    return ChatIdeaTool(
        handler=get_node_neighborhood,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=NodeNeighborhoodArgs,
    )
