from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VectorFallbackGraphContextProvider:
    def __init__(self, graph_context_provider, entity_retriever) -> None:
        self._graph_context_provider = graph_context_provider
        self._entity_retriever = entity_retriever

    def get_node_neighborhood(
        self,
        node_id: str,
        node_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        page: int = 1,
        depth: int = 1,
    ) -> dict:
        import inspect
        node_types = node_types or []
        relationship_types = relationship_types or []
        
        # 写代码时要加必要的中文注释：通过 inspect 检查底层 provider 是否支持 depth 参数，以保证对 Mock 的兼容
        sig = inspect.signature(self._graph_context_provider.get_node_neighborhood)
        kwargs = {}
        if "depth" in sig.parameters:
            kwargs["depth"] = depth

        context = self._graph_context_provider.get_node_neighborhood(
            node_id=node_id,
            node_types=node_types,
            relationship_types=relationship_types,
            page=page,
            **kwargs
        )
        if not _is_empty_lookup(context):
            return context

        # 精确和 alias 查询都为空时，用实体向量检索兜底，避免让模型继续猜节点 ID。
        candidates = self._entity_retriever.retrieve(node_id, top_k=1)
        if not candidates:
            logger.info("graph_tool_vector_fallback:miss requested_node_id=%s", node_id)
            return _with_lookup_note(
                context,
                {
                    "status": "vector_fallback_miss",
                    "requested_node_id": node_id,
                    "message": "未精确匹配到请求节点，且未找到可用的向量相似实体节点。",
                },
            )

        candidate = candidates[0]
        resolved_node_id = candidate.get("id")
        if not resolved_node_id:
            logger.info("graph_tool_vector_fallback:miss requested_node_id=%s reason=empty_candidate_id", node_id)
            return _with_lookup_note(
                context,
                {
                    "status": "vector_fallback_miss",
                    "requested_node_id": node_id,
                    "message": "未精确匹配到请求节点，且向量相似实体结果缺少可查询的实体 ID。",
                },
            )

        fallback_context = self._graph_context_provider.get_node_neighborhood(
            node_id=resolved_node_id,
            node_types=node_types,
            relationship_types=relationship_types,
            page=page,
            **kwargs
        )
        logger.info(
            "graph_tool_vector_fallback:resolved requested_node_id=%s resolved_node_id=%s resolved_entity_type=%s score=%s",
            node_id,
            resolved_node_id,
            candidate.get("entity_type"),
            candidate.get("score"),
        )
        return _with_lookup_note(
            fallback_context,
            {
                "status": "vector_fallback",
                "requested_node_id": node_id,
                "resolved_node_id": resolved_node_id,
                "resolved_entity_type": candidate.get("entity_type"),
                "score": candidate.get("score"),
                "message": "未精确匹配到请求节点，已采用向量相似度最高的非论文实体节点进行一跳查询。",
            },
        )


def _is_empty_lookup(context: dict) -> bool:
    if context.get("status") == "error":
        return False
    return not context.get("entities") and not context.get("relationships")


def _with_lookup_note(context: dict, note: dict) -> dict:
    updated = dict(context)
    existing_notes = list(updated.get("lookup_notes") or [])
    updated["lookup_notes"] = [*existing_notes, _clean_note(note)]
    return updated


def _clean_note(note: dict) -> dict:
    return {key: value for key, value in note.items() if value is not None}
