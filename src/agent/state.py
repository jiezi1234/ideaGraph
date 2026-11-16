from __future__ import annotations

from typing import NotRequired, Required, TypedDict

from pydantic import BaseModel

from src.schemas.agent import CandidatePaper, CanonicalIdea, IdeaComparison, IdeaEvaluation, IdeaOutput
from src.schemas.document import RetrievedContextChunk


class GraphContext(TypedDict):
    entities: list[dict]
    relationships: list[dict]
    page: int
    page_size: int
    total_entities: int
    total_relationships: int


class ToolTraceEntry(TypedDict):
    round: int
    tool_calls: list[dict]
    observation: GraphContext


class AgentState(TypedDict):
    user_query: Required[str]
    seed_paper_id: NotRequired[str]
    retrieval_query: NotRequired[str]
    retrieval_mode: NotRequired[str]
    query_keywords: NotRequired[list[str]]
    candidate_papers: NotRequired[list[CandidatePaper]]
    selected_paper: NotRequired[CandidatePaper]
    related_entities: NotRequired[list[dict]]
    retrieved_chunks: NotRequired[list[RetrievedContextChunk]]
    graph_context: NotRequired[GraphContext]
    tool_trace: NotRequired[list[ToolTraceEntry]]
    evaluation_graph_context: NotRequired[GraphContext]
    evaluation_tool_trace: NotRequired[list[ToolTraceEntry]]
    idea_output: NotRequired[BaseModel]
    initial_idea_output: NotRequired[BaseModel]
    canonical_idea: NotRequired[CanonicalIdea]
    generation_profile: NotRequired[str]
    prompt_variant: NotRequired[str]
    idea_evaluation: NotRequired[IdeaEvaluation]
    idea_comparison: NotRequired[IdeaComparison]
    idea_thinking: NotRequired[str]
    iteration: NotRequired[int]
    errors: NotRequired[list[str]]


class AgentStateUpdate(TypedDict, total=False):
    seed_paper_id: str
    retrieval_query: str
    retrieval_mode: str
    query_keywords: list[str]
    candidate_papers: list[CandidatePaper]
    selected_paper: CandidatePaper
    related_entities: list[dict]
    retrieved_chunks: list[RetrievedContextChunk]
    graph_context: GraphContext
    tool_trace: list[ToolTraceEntry]
    evaluation_graph_context: GraphContext
    evaluation_tool_trace: list[ToolTraceEntry]
    idea_output: BaseModel
    initial_idea_output: BaseModel
    canonical_idea: CanonicalIdea
    generation_profile: str
    prompt_variant: str
    idea_evaluation: IdeaEvaluation
    idea_comparison: IdeaComparison
    idea_thinking: str
    iteration: int
    errors: list[str]


class QueryUnderstandingState(TypedDict):
    user_query: Required[str]


class PaperSelectionState(TypedDict):
    user_query: Required[str]
    candidate_papers: Required[list[CandidatePaper]]


class IdeaEvaluationState(TypedDict):
    idea_output: Required[IdeaOutput]
    user_query: NotRequired[str]
    candidate_papers: NotRequired[list[CandidatePaper]]
    selected_paper: NotRequired[CandidatePaper]
    graph_context: NotRequired[GraphContext]
