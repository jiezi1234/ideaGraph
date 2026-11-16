from __future__ import annotations

from src.agent.graph import build_idea_chat_graph
from src.observability.langfuse import flush_langfuse_client
from src.observability.langfuse import summarize_idea_result
from src.retrieval.base import PaperRetriever
from src.retrieval.selector import TopScorePaperSelector


class IdeaChatWorkflow:
    def __init__(
        self,
        retriever: PaperRetriever,
        selector: TopScorePaperSelector,
        model,
        retrieval_top_k: int,
        evaluation_model=None,
        graph_context_provider=None,
        initial_entity_retriever=None,
        context_chunk_retriever=None,
        seed_paper_provider=None,
        query_understanding_prompt_template: str | None = None,
        agent_prompt_template: str | None = None,
        evaluation_prompt_template: str | None = None,
        generation_profile: str = "ideagraph",
        retrieval_mode: str = "graph",
        langfuse_client=None,
    ) -> None:
        self._langfuse_client = langfuse_client
        self._retrieval_mode = retrieval_mode
        self._graph = build_idea_chat_graph(
            retriever=retriever,
            selector=selector,
            model=model,
            retrieval_top_k=retrieval_top_k,
            evaluation_model=evaluation_model,
            graph_context_provider=graph_context_provider,
            initial_entity_retriever=initial_entity_retriever,
            context_chunk_retriever=context_chunk_retriever,
            seed_paper_provider=seed_paper_provider,
            query_understanding_prompt_template=query_understanding_prompt_template,
            agent_prompt_template=agent_prompt_template,
            evaluation_prompt_template=evaluation_prompt_template,
            generation_profile=generation_profile,
            retrieval_mode=retrieval_mode,
            langfuse_client=langfuse_client,
        )

    def run(self, user_query: str, seed_paper_id: str | None = None) -> dict:
        input_state = {"user_query": user_query, "retrieval_mode": self._retrieval_mode}
        if seed_paper_id:
            input_state["seed_paper_id"] = seed_paper_id
        try:
            if self._langfuse_client is None:
                result = dict(self._graph.invoke(input_state))
                return _strip_internal_chat_state(result)
            with self._langfuse_client.start_as_current_observation(
                name="chat-idea",
                as_type="agent",
                input=input_state,
                metadata={"workflow": "idea_chat", "stage": "root", "has_seed_paper": bool(seed_paper_id)},
            ) as span:
                result = dict(self._graph.invoke(input_state))
                result = _strip_internal_chat_state(result)
                span.update(
                    input=input_state,
                    output=summarize_idea_result(result),
                    metadata={
                        "workflow": "idea_chat",
                        "stage": "root",
                        "seed_paper_id": result.get("seed_paper_id"),
                        "has_seed_paper": bool(result.get("seed_paper_id")),
                        "retrieval_query": result.get("retrieval_query"),
                        "query_keywords": result.get("query_keywords", []),
                        "has_graph_context": bool(result.get("graph_context")),
                        "has_idea_evaluation": bool(result.get("idea_evaluation")),
                    },
                )
                return result
        finally:
            flush_langfuse_client(self._langfuse_client)


def _strip_internal_chat_state(result: dict) -> dict:
    result.pop("iteration", None)
    return result
