import logging

from src.schemas.agent import AIIdeaBenchIdeaOutput
from src.schemas.agent import CandidatePaper, CriterionRanking, IdeaComparison, IdeaEvaluation, IdeaOutput
from src.schemas.document import RetrievedContextChunk, RetrievedPaperDocument


def _evaluation() -> IdeaEvaluation:
    criterion = {
        "score": 4,
        "comment": "清楚。",
        "revision_suggestion": "继续收窄实验。",
    }
    return IdeaEvaluation(
        overall_score=4,
        summary="方向可行。",
        novelty=criterion,
        clarity=criterion,
        feasibility=criterion,
        relevance=criterion,
        expected_effectiveness=criterion,
        evidence_grounding=criterion,
        strengths=["图谱上下文明确"],
        major_issues=["实验还需细化"],
        revision_instruction="补充实验协议。",
    )


def _comparison() -> IdeaComparison:
    return IdeaComparison(
        overall_first="idea_b",
        overall_explanation="Idea B 在多个维度上更优。",
        novelty=CriterionRanking(first="idea_b", explanation="Idea B 更具新意。"),
        clarity=CriterionRanking(first="tie", explanation="两者清晰度相近。"),
        feasibility=CriterionRanking(first="idea_b", explanation="Idea B 更可行。"),
        relevance=CriterionRanking(first="tie", explanation="两者相关性相当。"),
        expected_effectiveness=CriterionRanking(first="idea_b", explanation="Idea B 预期效果更好。"),
        evidence_grounding=CriterionRanking(first="idea_b", explanation="Idea B 证据更充分。"),
    )


class FakeStructuredResult:
    def __init__(self, parsed, reasoning: str = "", usage: dict | None = None):
        self.parsed = parsed
        self.reasoning = reasoning
        self.usage = usage or {"input": 11, "output": 7, "total": 18}


class FakeToolCall:
    def __init__(self, tool_name: str, arguments: dict, tool_call_id: str = "call_1"):
        self.tool_name = tool_name
        self.arguments = arguments
        self.tool_call_id = tool_call_id


class FakeChatResult:
    def __init__(
        self,
        content: str = "",
        tool_calls: list[FakeToolCall] | None = None,
        finish_reason: str = "stop",
        reasoning: str = "",
        usage: dict | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.reasoning = reasoning
        self.usage = usage or {"input": 11, "output": 7, "total": 18}


class FakeChatModel:
    def __init__(self, structured_outputs, tool_responses=None):
        self.structured_outputs = list(structured_outputs)
        self.tool_responses = list(tool_responses or [])
        self.structured_calls = []
        self.tool_calls = []

    def structured_json(self, messages, schema, name: str, metadata: dict | None = None):
        self.structured_calls.append((schema, list(messages), name, metadata))
        return FakeStructuredResult(self.structured_outputs.pop(0))

    def chat_with_tools(self, messages, tools, name: str, metadata: dict | None = None):
        self.tool_calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "name": name,
                "metadata": metadata,
            }
        )
        return self.tool_responses.pop(0)


class StubRetriever:
    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = []

    def retrieve(self, query: str, top_k: int) -> list[CandidatePaper]:
        self.calls.append((query, top_k))
        return self._candidates


class StubChunkRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedContextChunk]:
        self.calls.append((query, top_k))
        return self._chunks[:top_k]


class StubSelector:
    def select(self, user_query: str, candidates: list[CandidatePaper]) -> CandidatePaper:
        del user_query
        return candidates[0]


class StubSeedPaperProvider:
    def __init__(self, paper: RetrievedPaperDocument | None):
        self._paper = paper
        self.calls = []

    def get_paper_by_id(self, paper_id: str) -> RetrievedPaperDocument | None:
        self.calls.append(paper_id)
        return self._paper


class FakeLangfuseSpan:
    def __init__(self, client, name: str, as_type: str, input_payload, metadata):
        self._client = client
        self.name = name
        self.as_type = as_type
        self.input = input_payload
        self.metadata = metadata
        self.updates = []

    def __enter__(self):
        self._client.observations.append(
            {
                "name": self.name,
                "as_type": self.as_type,
                "input": self.input,
                "metadata": self.metadata,
                "updates": self.updates,
            }
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuseClient:
    def __init__(self):
        self.observations = []

    def start_as_current_observation(self, name, as_type="span", input=None, metadata=None):
        return FakeLangfuseSpan(self, name=name, as_type=as_type, input_payload=input, metadata=metadata)


def test_idea_chat_workflow_uses_openai_structured_output_without_tools():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="paper-1",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    idea = IdeaOutput(
        title="Retriever-guided research copilot",
        summary="Use retrieved citation evidence to scaffold ideas.",
        problem="Idea generation lacks grounded paper context.",
        method="Retrieve, rank, and synthesize from relevant papers.",
        core_paper_id="paper-1",
        core_paper_title="Graph-guided discovery",
    )
    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph", "idea generation"], "retrieval_query": "graph idea generation"},
            idea,
        ]
    )
    retriever = StubRetriever([candidate])
    workflow = IdeaChatWorkflow(
        retriever=retriever,
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
    )

    result = workflow.run("  graph idea generation  ")

    assert result["retrieval_query"] == "graph idea generation"
    assert result["candidate_papers"] == [candidate]
    assert result["idea_output"] == idea
    assert retriever.calls == [("graph idea generation", 3)]
    assert [call[0].__name__ for call in model.structured_calls] == ["QueryUnderstandingOutput", "IdeaOutput"]
    assert [call[2] for call in model.structured_calls] == ["query_understanding", "final_idea_generation"]


def test_idea_chat_workflow_paper_vector_mode_disables_graph_tools():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="paper-1",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    idea = IdeaOutput(
        title="Vector-only idea",
        summary="Use paper vector retrieval without graph tools.",
        problem="A fair baseline must not receive graph context.",
        method="Retrieve candidate papers and generate directly from paper text.",
        core_paper_id="paper-1",
        core_paper_title="Graph-guided discovery",
    )

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            raise AssertionError("paper_vector mode must not call graph tools")

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            FakeChatResult(
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "Graph"})]
            )
        ],
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=GraphContextProvider(),
        retrieval_mode="paper_vector",
    )

    result = workflow.run("graph idea generation")

    assert result["retrieval_mode"] == "paper_vector"
    assert "tool_trace" not in result
    assert "graph_context" not in result
    assert model.tool_calls == []
    assert result["idea_output"] == idea


def test_idea_chat_workflow_chunk_vector_mode_injects_retrieved_chunks():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="paper-1",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    chunk = RetrievedContextChunk(
        paper_id="paper-1",
        title="Graph-guided discovery",
        chunk_id="chunk-1",
        chunk_index=0,
        paragraph_start=0,
        paragraph_end=1,
        score=0.82,
        text_source="full_text_md",
        text="Title: Graph-guided discovery\nChunk:\nDetailed paragraph evidence.",
    )
    idea = IdeaOutput(
        title="Chunk vector idea",
        summary="Use retrieved full-text chunks without graph tools.",
        problem="Abstract-only retrieval can miss detailed evidence.",
        method="Retrieve chunks and generate a grounded idea from vector evidence.",
        core_paper_id="paper-1",
        core_paper_title="Graph-guided discovery",
    )
    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ]
    )
    chunk_retriever = StubChunkRetriever([chunk])
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        context_chunk_retriever=chunk_retriever,
        retrieval_mode="chunk_vector",
    )

    result = workflow.run("graph idea generation")

    final_messages = model.structured_calls[-1][1]
    assert chunk_retriever.calls == [("graph", 3)]
    assert result["retrieval_mode"] == "chunk_vector"
    assert result["retrieved_chunks"] == [chunk]
    assert "tool_trace" not in result
    assert "Retrieved vector context" in final_messages[1]["content"]
    assert "Detailed paragraph evidence" in final_messages[1]["content"]
    assert "Score: 0.82" in final_messages[1]["content"]
    assert result["idea_output"] == idea


def test_idea_chat_workflow_uses_seed_paper_without_query_understanding_or_retrieval():
    from src.workflows.idea_chat import IdeaChatWorkflow

    seed_paper = RetrievedPaperDocument(
        document_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        year=2024,
        cited_by_count=12,
    )
    idea = IdeaOutput(
        title="Seeded graph idea",
        summary="Start from an explicit graph paper instead of embedding retrieval.",
        problem="The user already knows which paper should ground the idea.",
        method="Load the paper by graph id and continue directly into generation.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )
    model = FakeChatModel(structured_outputs=[idea])
    retriever = StubRetriever(
        [
            CandidatePaper(
                paper_id="unused",
                title="Unused",
                abstract="This candidate should never be retrieved.",
            )
        ]
    )
    seed_provider = StubSeedPaperProvider(seed_paper)
    workflow = IdeaChatWorkflow(
        retriever=retriever,
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        seed_paper_provider=seed_provider,
    )

    result = workflow.run(
        "generate an idea grounded in this paper",
        seed_paper_id="Graph-guided discovery",
    )

    assert seed_provider.calls == ["Graph-guided discovery"]
    assert retriever.calls == []
    assert result["seed_paper_id"] == "Graph-guided discovery"
    assert result["retrieval_query"] == "generate an idea grounded in this paper"
    assert result["query_keywords"] == []
    assert result["candidate_papers"] == [
        CandidatePaper(
            paper_id="Graph-guided discovery",
            title="Graph-guided discovery",
            abstract="Graph retrieval can improve idea generation.",
            cited_by_count=12,
            score=1.0,
        )
    ]
    assert result["selected_paper"] == result["candidate_papers"][0]
    assert result["idea_output"] == idea
    assert [call[0].__name__ for call in model.structured_calls] == ["IdeaOutput"]
    assert [call[2] for call in model.structured_calls] == ["final_idea_generation"]


def test_idea_chat_workflow_uses_ai_idea_bench_profile_schema_and_canonical_output():
    from src.workflows.idea_chat import IdeaChatWorkflow

    seed_paper = RetrievedPaperDocument(
        document_id="LiDAR reference",
        title="LiDAR reference",
        abstract="A reference paper about LiDAR perception.",
        year=2024,
        cited_by_count=12,
    )
    idea = AIIdeaBenchIdeaOutput(
        Name="lidar_bev_uncertainty",
        Title="Uncertainty-aware BEV fusion for LiDAR perception",
        Motivation="Current LiDAR perception systems miss uncertainty from sparse point clouds.",
        Experiment="Train a BEV fusion model and evaluate detection robustness under adverse weather.",
    )
    model = FakeChatModel(structured_outputs=[idea])
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        seed_paper_provider=StubSeedPaperProvider(seed_paper),
        evaluation_prompt_template="Evaluation Agent",
        generation_profile="ai_idea_bench",
    )

    result = workflow.run("generate an idea", seed_paper_id="LiDAR reference")

    assert result["idea_output"] == idea
    assert result["generation_profile"] == "ai_idea_bench"
    assert result["prompt_variant"] == "ai_idea_bench_v1"
    assert result["canonical_idea"].title == idea.Title
    assert result["canonical_idea"].motivation == idea.Motivation
    assert result["canonical_idea"].experiment == idea.Experiment
    assert [call[0].__name__ for call in model.structured_calls] == ["AIIdeaBenchIdeaOutput"]
    assert [call[2] for call in model.structured_calls] == ["final_idea_generation"]
    assert "idea_evaluation" not in result


def test_idea_chat_workflow_returns_error_when_seed_paper_is_missing():
    from src.workflows.idea_chat import IdeaChatWorkflow

    model = FakeChatModel(structured_outputs=[])
    retriever = StubRetriever([])
    seed_provider = StubSeedPaperProvider(None)
    workflow = IdeaChatWorkflow(
        retriever=retriever,
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        seed_paper_provider=seed_provider,
    )

    result = workflow.run("generate an idea", seed_paper_id="Missing Paper")

    assert seed_provider.calls == ["Missing Paper"]
    assert retriever.calls == []
    assert model.structured_calls == []
    assert result == {
        "user_query": "generate an idea",
        "retrieval_mode": "graph",
        "seed_paper_id": "Missing Paper",
        "candidate_papers": [],
        "errors": ["Seed paper not found in graph: Missing Paper"],
    }


def test_idea_chat_workflow_runs_openai_tool_loop_and_archives_trace(caplog):
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper"},
            {"id": "knowledge graph", "node_type": "Concept"},
            {"id": "concept_c", "node_type": "Concept"},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            },
            {
                "source_id": "knowledge graph",
                "target_id": "concept_c",
                "relationship_type": "RELATED",
            }
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 3,
        "total_relationships": 2,
    }
    idea = IdeaOutput(
        title="OpenAI tool-call ideation",
        summary="Use LangGraph tool observations to ground ideas.",
        problem="Retrieved papers lack surrounding graph context.",
        method="Call graph tools through OpenAI-compatible tool calls and synthesize the result.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )

    class GraphContextProvider:
        def __init__(self):
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append((node_id, node_types, relationship_types, page))
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "relationship_types": ["CONTAINS"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(content="Enough context."),
        ],
    )
    provider = GraphContextProvider()
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=provider,
        langfuse_client=FakeLangfuseClient(),
    )

    with caplog.at_level(logging.INFO):
        result = workflow.run("graph idea generation")

    assert provider.calls == [
        ("Graph-guided discovery", [], ["CONTAINS"], 1),
        ("knowledge graph", [], [], 1),
        ("concept_c", [], [], 1),
    ]
    assert result["graph_context"]["entities"] == graph_context["entities"]
    assert result["tool_trace"][0]["tool_calls"][0]["tool_name"] == "get_node_neighborhood"
    assert result["tool_trace"][0]["tool_calls"][0]["arguments"]["node_id"] == "Graph-guided discovery"
    assert result["tool_trace"][1]["tool_calls"][0]["arguments"]["node_id"] == "knowledge graph"
    assert result["tool_trace"][2]["tool_calls"][0]["arguments"]["node_id"] == "concept_c"
    assert result["idea_output"] == idea
    assert model.tool_calls[0]["tools"][0]["function"]["name"] == "get_node_neighborhood"
    assert model.tool_calls[0]["name"] == "agent_generation_round_0"
    assert "graph_tool_planning:planned count=1" in caplog.text
    assert "graph_tool_call:done tool=get_node_neighborhood" in caplog.text
    assert any(observation["name"].startswith("graph_tool_execution_round_") for observation in workflow._langfuse_client.observations)


def test_idea_chat_workflow_uses_direct_final_idea_from_last_tool_round():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper"},
            {"id": "knowledge graph", "node_type": "Concept"},
            {"id": "concept_c", "node_type": "Concept"},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            },
            {
                "source_id": "knowledge graph",
                "target_id": "concept_c",
                "relationship_type": "RELATED",
            }
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 3,
        "total_relationships": 2,
    }
    idea = IdeaOutput(
        title="Direct final JSON idea",
        summary="Use the last assistant response as the final structured idea.",
        problem="The workflow regenerates the final idea after the tool loop has already finished.",
        method="Parse the last no-tool assistant response directly when it already contains IdeaOutput JSON.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            del node_types, relationship_types, page
            assert node_id in ("Graph-guided discovery", "knowledge graph", "concept_c")
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "relationship_types": ["CONTAINS"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(content=idea.model_dump_json()),
        ],
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=GraphContextProvider(),
    )

    result = workflow.run("graph idea generation")

    assert result["idea_output"] == idea
    assert result["graph_context"]["entities"] == graph_context["entities"]
    assert [call[0].__name__ for call in model.structured_calls] == ["QueryUnderstandingOutput"]
    assert [call["name"] for call in model.tool_calls] == [
        "agent_generation_round_0",
        "agent_generation_round_1",
        "agent_generation_round_2",
        "agent_generation_round_3",
    ]


def test_idea_chat_workflow_extracts_direct_final_json_after_required_tool_path():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    idea = IdeaOutput(
        title="Prefixed final JSON idea",
        summary="Extract the JSON object even when the model adds a short preface.",
        problem="Some models explain before returning structured JSON.",
        method="Find the first valid JSON payload in the final assistant content.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper"},
            {"id": "knowledge graph", "node_type": "Concept"},
            {"id": "concept_c", "node_type": "Concept"},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            },
            {
                "source_id": "knowledge graph",
                "target_id": "concept_c",
                "relationship_type": "RELATED",
            },
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 3,
        "total_relationships": 2,
    }

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            del node_types, relationship_types, page
            assert node_id in ("Graph-guided discovery", "knowledge graph", "concept_c")
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "relationship_types": ["CONTAINS"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(content=f"下面是最终方案：\n{idea.model_dump_json()}\n可直接使用。"),
        ],
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=GraphContextProvider(),
    )

    result = workflow.run("graph idea generation")

    assert result["idea_output"] == idea
    assert [call[0].__name__ for call in model.structured_calls] == ["QueryUnderstandingOutput"]
    assert [call["name"] for call in model.tool_calls] == [
        "agent_generation_round_0",
        "agent_generation_round_1",
        "agent_generation_round_2",
        "agent_generation_round_3",
    ]


def test_final_idea_generation_fallback_user_message_is_minimal():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    idea = IdeaOutput(
        title="Fallback structured idea",
        summary="Use structured output only when direct response parsing fails.",
        problem="The fallback prompt previously repeated retrieval and graph context.",
        method="Ask the model to provide the final idea without restating all context.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper"},
            {"id": "knowledge graph", "node_type": "Concept"},
            {"id": "concept_c", "node_type": "Concept"},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            },
            {
                "source_id": "knowledge graph",
                "target_id": "concept_c",
                "relationship_type": "RELATED",
            },
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 3,
        "total_relationships": 2,
    }

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            del node_types, relationship_types, page
            assert node_id in ("Graph-guided discovery", "knowledge graph", "concept_c")
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "relationship_types": ["CONTAINS"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(content="我还没整理成 JSON。"),
        ],
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=GraphContextProvider(),
    )

    result = workflow.run("graph idea generation")

    assert result["idea_output"] == idea
    final_messages = model.structured_calls[1][1]
    final_user_content = final_messages[-1]["content"]
    assert "Current phase: final_idea_generation" in final_user_content
    assert "Generate the final idea now" in final_user_content
    assert "Merged graph context:" not in final_user_content
    assert "Related papers:" not in final_user_content
    assert "Retrieval query:" not in final_user_content


def test_merge_graph_contexts_preserves_lookup_notes():
    from src.agent.graph import _merge_graph_contexts

    merged = _merge_graph_contexts(
        [
            {
                "entities": [{"id": "neat", "entity_type": "Method"}],
                "relationships": [],
                "lookup_notes": [
                    {
                        "status": "vector_fallback",
                        "requested_node_id": "进化神经网络",
                        "resolved_node_id": "neat",
                    }
                ],
                "total_entities": 1,
                "total_relationships": 0,
            }
        ]
    )

    assert merged["lookup_notes"] == [
        {
            "status": "vector_fallback",
            "requested_node_id": "进化神经网络",
            "resolved_node_id": "neat",
        }
    ]


def test_idea_chat_workflow_returns_tool_errors_to_model_for_retry(caplog):
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper"},
            {"id": "knowledge graph", "node_type": "Concept"},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            }
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 2,
        "total_relationships": 1,
    }
    idea = IdeaOutput(
        title="Retry after tool error",
        summary="The model retries tool calls after receiving a structured error.",
        problem="Invalid tool arguments used to abort the workflow before the model could recover.",
        method="Return tool errors as tool messages so the next round can correct the call.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )

    class GraphContextProvider:
        def __init__(self):
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append((node_id, node_types, relationship_types, page))
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "node_types": ["Metric"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "node_types": ["Paper"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(content="Enough context."),
        ],
    )
    provider = GraphContextProvider()
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=provider,
    )

    with caplog.at_level(logging.WARNING):
        result = workflow.run("graph idea generation")

    retry_messages = model.tool_calls[1]["messages"]
    tool_error_message = next(message for message in retry_messages if message["role"] == "tool")
    assert '"status": "error"' in tool_error_message["content"]
    assert "Metric" in tool_error_message["content"]
    assert "node_types must be one of" in tool_error_message["content"]
    assert provider.calls == [
        ("Graph-guided discovery", ["Paper"], [], 1),
        ("knowledge graph", [], [], 1),
    ]
    assert result["graph_context"]["entities"] == graph_context["entities"]
    assert "graph_tool_call:error tool=get_node_neighborhood" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_idea_chat_workflow_records_round_reasoning_in_langfuse_when_content_is_empty():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper"},
            {"id": "knowledge graph", "node_type": "Concept"},
            {"id": "concept_c", "node_type": "Concept"},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            },
            {
                "source_id": "knowledge graph",
                "target_id": "concept_c",
                "relationship_type": "RELATED",
            }
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 3,
        "total_relationships": 2,
    }
    idea = IdeaOutput(
        title="Fallback final idea",
        summary="Use a separate final generation step when the last tool round has no JSON content.",
        problem="The last tool-planning round may spend tokens on reasoning without emitting visible content.",
        method="Record the round response in Langfuse, then fall back to final structured generation if needed.",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            del node_types, relationship_types, page
            assert node_id in ("Graph-guided discovery", "knowledge graph", "concept_c")
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery"},
                    )
                ],
                usage={"input": 12, "output": 8, "total": 20},
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(
                content="",
                finish_reason="stop",
                reasoning="I have enough graph evidence; now I am drafting the final idea internally.",
                usage={"input": 100, "output": 220, "total": 320},
            ),
        ],
    )
    langfuse_client = FakeLangfuseClient()
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=GraphContextProvider(),
        langfuse_client=langfuse_client,
    )

    result = workflow.run("graph idea generation")

    assert result["idea_output"] == idea
    round_observation = next(
        observation
        for observation in langfuse_client.observations
        if observation["name"] == "agent_generation_round_3_result"
    )
    assert round_observation["updates"][-1]["output"]["content"] == ""
    assert round_observation["updates"][-1]["output"]["reasoning"] == (
        "I have enough graph evidence; now I am drafting the final idea internally."
    )
    assert round_observation["updates"][-1]["output"]["finish_reason"] == "stop"


def test_idea_chat_workflow_returns_structured_error_when_no_candidates():
    from src.workflows.idea_chat import IdeaChatWorkflow

    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int) -> list[CandidatePaper]:
            del query, top_k
            return []

    class UnusedSelector:
        def select(self, user_query: str, candidates: list[CandidatePaper]) -> CandidatePaper:
            raise AssertionError("selector should not be called when no candidates are returned")

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["novel graph ideas"], "retrieval_query": "novel graph ideas"},
        ]
    )
    workflow = IdeaChatWorkflow(
        retriever=EmptyRetriever(),
        selector=UnusedSelector(),
        model=model,
        retrieval_top_k=5,
    )

    result = workflow.run("novel graph ideas")

    assert result == {
        "user_query": "novel graph ideas",
        "retrieval_mode": "graph",
        "retrieval_query": "novel graph ideas",
        "query_keywords": ["novel graph ideas"],
        "candidate_papers": [],
        "errors": ["No candidate papers retrieved for query: novel graph ideas"],
    }


def test_idea_chat_workflow_runs_openai_structured_evaluation():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="paper-1",
        title="Knowledge graph ideation",
        abstract="Knowledge graphs support scientific idea generation.",
        score=0.91,
    )
    idea = IdeaOutput(
        title="Graph-based idea mining",
        summary="Use knowledge graphs to mine grounded ideas.",
        problem="Idea generation lacks structured retrieval context.",
        method="Extract query concepts and retrieve matching papers.",
        core_paper_id="paper-1",
        core_paper_title="Knowledge graph ideation",
    )
    evaluation = _evaluation()
    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["knowledge graph"], "retrieval_query": "knowledge graph"},
            idea,
            evaluation,
            idea,
            evaluation,
            _comparison(),
        ]
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        evaluation_prompt_template="Evaluation Agent",
    )

    result = workflow.run("graph-based scientific idea generation")

    assert result["idea_output"] == idea
    assert result["idea_evaluation"] == evaluation
    assert [call[0].__name__ for call in model.structured_calls] == [
        "QueryUnderstandingOutput",
        "IdeaOutput",
        "IdeaEvaluation",
        "IdeaOutput",
        "IdeaEvaluation",
        "IdeaComparison",
    ]
    assert [call[2] for call in model.structured_calls] == [
        "query_understanding",
        "final_idea_generation",
        "idea_evaluation",
        "final_idea_generation",
        "idea_evaluation",
        "idea_comparison",
    ]


def test_idea_chat_workflow_can_use_separate_evaluation_model():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="paper-1",
        title="Knowledge graph ideation",
        abstract="Knowledge graphs support scientific idea generation.",
        score=0.91,
    )
    initial_idea = IdeaOutput(
        title="Graph-based idea mining",
        summary="Use knowledge graphs to mine grounded ideas.",
        problem="Idea generation lacks structured retrieval context.",
        method="Extract query concepts and retrieve matching papers.",
        core_paper_id="paper-1",
        core_paper_title="Knowledge graph ideation",
    )
    improved_idea = IdeaOutput(
        title="Graph-based idea mining with explicit evidence checks",
        summary="Use evaluation feedback to make graph evidence checks explicit.",
        problem="Idea generation lacks structured retrieval context.",
        method="Retrieve graph evidence, generate an idea, then refine against evaluator feedback.",
        core_paper_id="paper-1",
        core_paper_title="Knowledge graph ideation",
    )
    generation_model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["knowledge graph"], "retrieval_query": "knowledge graph"},
            initial_idea,
            improved_idea,
            _comparison(),
        ]
    )
    evaluation_model = FakeChatModel(
        structured_outputs=[
            _evaluation(),
            _evaluation(),
        ]
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=generation_model,
        evaluation_model=evaluation_model,
        retrieval_top_k=3,
        evaluation_prompt_template="Evaluation Agent",
    )

    result = workflow.run("graph-based scientific idea generation")

    assert result["idea_output"] == improved_idea
    assert [call[2] for call in generation_model.structured_calls] == [
        "query_understanding",
        "final_idea_generation",
        "final_idea_generation",
        "idea_comparison",
    ]
    assert [call[2] for call in evaluation_model.structured_calls] == [
        "idea_evaluation",
        "idea_evaluation",
    ]


def test_idea_chat_workflow_runs_one_improvement_pass_after_evaluation():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="paper-1",
        title="Knowledge graph ideation",
        abstract="Knowledge graphs support scientific idea generation.",
        score=0.91,
    )
    initial_idea = IdeaOutput(
        title="初稿想法",
        summary="先产出一个初稿方案。",
        problem="初稿问题定义还比较宽。",
        method="先给出基础方法。",
        core_paper_id="paper-1",
        core_paper_title="Knowledge graph ideation",
    )
    initial_evaluation = _evaluation()
    improved_idea = IdeaOutput(
        title="改进后想法",
        summary="根据评审反馈收窄并强化实验设计。",
        problem="聚焦一个更明确的问题边界。",
        method="在原始方案上加入更清晰的模块划分和实验协议。",
        core_paper_id="paper-1",
        core_paper_title="Knowledge graph ideation",
    )
    improved_evaluation = IdeaEvaluation(
        overall_score=5,
        summary="改进后更完整。",
        novelty={"score": 5, "comment": "更聚焦。", "revision_suggestion": "保持。"},
        clarity={"score": 5, "comment": "表达更清楚。", "revision_suggestion": "保持。"},
        feasibility={"score": 4, "comment": "可执行。", "revision_suggestion": "补充资源预算。"},
        relevance={"score": 5, "comment": "主题贴合。", "revision_suggestion": "保持。"},
        expected_effectiveness={"score": 4, "comment": "更有说服力。", "revision_suggestion": "补充基线。"},
        evidence_grounding={"score": 4, "comment": "依据更充分。", "revision_suggestion": "补更多引用证据。"},
        strengths=["问题更聚焦"],
        major_issues=["还可补充资源预算"],
        revision_instruction="保持当前收窄方向，并补充资源约束。",
    )
    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["knowledge graph"], "retrieval_query": "knowledge graph"},
            initial_idea,
            initial_evaluation,
            improved_idea,
            improved_evaluation,
            _comparison(),
        ]
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        evaluation_prompt_template="Evaluation Agent",
    )

    result = workflow.run("graph-based scientific idea generation")

    assert result["idea_output"] == improved_idea
    assert result["idea_evaluation"] == improved_evaluation
    assert result["idea_comparison"] is not None
    assert "iteration" not in result
    assert result["initial_idea_output"] == initial_idea
    assert "initial_evaluation" not in result
    assert [call[0].__name__ for call in model.structured_calls] == [
        "QueryUnderstandingOutput",
        "IdeaOutput",
        "IdeaEvaluation",
        "IdeaOutput",
        "IdeaEvaluation",
        "IdeaComparison",
    ]
    second_generation_messages = model.structured_calls[3][1]
    assert second_generation_messages[1]["content"].find(initial_idea.title) != -1
    assert second_generation_messages[1]["content"].find(initial_evaluation.revision_instruction) != -1


def test_improvement_round_includes_previous_graph_context_and_trace():
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Graph-guided discovery",
        title="Graph-guided discovery",
        abstract="Graph retrieval can improve idea generation.",
        score=0.91,
    )
    graph_context = {
        "entities": [
            {"id": "Graph-guided discovery", "node_type": "Paper", "abstract": "Graph retrieval paper."},
            {"id": "knowledge graph", "entity_type": "Concept", "description": "Structured graph evidence."},
            {"id": "concept_c", "entity_type": "Concept", "description": "Additional concept."},
        ],
        "relationships": [
            {
                "source_id": "Graph-guided discovery",
                "target_id": "knowledge graph",
                "relationship_type": "CONTAINS",
            },
            {
                "source_id": "knowledge graph",
                "target_id": "concept_c",
                "relationship_type": "RELATED",
            }
        ],
        "page": 1,
        "page_size": 20,
        "total_entities": 3,
        "total_relationships": 2,
    }
    initial_idea = IdeaOutput(
        title="初稿想法",
        summary="先产出一个初稿方案。",
        problem="初稿问题定义还比较宽。",
        method="先给出基础方法。",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )
    initial_evaluation = _evaluation()
    improved_idea = IdeaOutput(
        title="改进后想法",
        summary="根据上一轮图谱观察和评审反馈改进。",
        problem="围绕知识图谱证据收窄问题。",
        method="继承上一轮图谱观察并补充改进方案。",
        core_paper_id="Graph-guided discovery",
        core_paper_title="Graph-guided discovery",
    )
    improved_evaluation = _evaluation()

    class GraphContextProvider:
        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            del node_types, relationship_types, page
            assert node_id in ("Graph-guided discovery", "knowledge graph", "concept_c")
            return graph_context

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["knowledge graph"], "retrieval_query": "knowledge graph"},
            initial_idea,
            initial_evaluation,
            improved_evaluation,
            _comparison(),
        ],
        tool_responses=[
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery", "relationship_types": ["CONTAINS"]},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(content="Enough context."),
            FakeChatResult(content="Evaluation has enough graph context."),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "knowledge graph"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "Graph-guided discovery"},
                    )
                ],
            ),
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        tool_name="get_node_neighborhood",
                        arguments={"node_id": "concept_c"},
                    )
                ],
            ),
            FakeChatResult(content=improved_idea.model_dump_json()),
            FakeChatResult(content="Improved evaluation has enough graph context."),
        ],
    )
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=GraphContextProvider(),
        evaluation_prompt_template="Evaluation Agent",
    )

    result = workflow.run("graph-based scientific idea generation")

    second_round_messages = next(
        call["messages"]
        for call in model.tool_calls
        if call["name"] == "agent_generation_round_0"
        and any(
            message["role"] == "user"
            and "Previous graph context from the initial generation" in message["content"]
            for message in call["messages"]
        )
    )
    second_round_user = next(m["content"] for m in second_round_messages if m["role"] == "user")
    assert "Previous graph context from the initial generation" in second_round_user
    assert "knowledge graph" in second_round_user
    assert "Graph-guided discovery -[CONTAINS]-> knowledge graph" in second_round_user
    assert "Previous graph tool trace from the initial generation" in second_round_user
    assert "get_node_neighborhood" in second_round_user
    assert '"relationship_types": ["CONTAINS"]' in second_round_user
    # 验证已删除工具消息回放（方法A），改用纯文本摘要（方法B）传递上一轮图谱上下文
    tool_messages = [m for m in second_round_messages if m["role"] == "tool"]
    assert len(tool_messages) == 0, "方法A（tool message replay）已移除，改进轮不应包含 tool role messages"
    assert result["idea_output"] == improved_idea
    assert result["idea_comparison"] is not None
    assert result["graph_context"]["entities"] == graph_context["entities"]
    assert result["tool_trace"][0]["tool_calls"][0]["arguments"]["node_id"] == "Graph-guided discovery"


def test_build_idea_chat_workflow_wires_openai_ollama_model(monkeypatch, tmp_path):
    from src.app import bootstrap
    from src.config.settings import Settings

    calls = {}

    class FakeRepository:
        @classmethod
        def from_connection(cls, uri: str, username: str, password: str, database: str):
            calls["repository"] = {
                "uri": uri,
                "username": username,
                "password": password,
                "database": database,
            }
            return {"repository": "instance"}

    class FakeEmbeddingModel:
        def __init__(self, base_url: str, model_name: str, timeout: float) -> None:
            calls["embedder"] = {
                "base_url": base_url,
                "model_name": model_name,
                "timeout": timeout,
            }

    def fake_build_chroma_client(database_name: str):
        calls["chroma_client_database"] = database_name
        return {"chroma": "client"}

    class FakeChromaVectorIndex:
        def __init__(
            self,
            client,
            database_name: str,
            embedder,
            tokenizer=None,
            chunk_min_tokens=0,
            chunk_max_tokens=0,
            chunk_overlap_tokens=0,
        ) -> None:
            calls["vector_index_instance"] = self
            calls["vector_index"] = {
                "client": client,
                "database_name": database_name,
                "embedder": embedder,
                "tokenizer": tokenizer,
                "chunk_min_tokens": chunk_min_tokens,
                "chunk_max_tokens": chunk_max_tokens,
                "chunk_overlap_tokens": chunk_overlap_tokens,
            }

    class FakeHuggingFaceChunkTokenizer:
        @classmethod
        def from_model_path(cls, model_path: str, *, local_files_only: bool = True):
            calls["chunk_tokenizer"] = {
                "model_path": model_path,
                "local_files_only": local_files_only,
            }
            return {"tokenizer": "instance"}

    class FakeRetriever:
        def __init__(self, repository, index) -> None:
            calls["retriever"] = {
                "repository": repository,
                "index": index,
            }

    class FakeEntityRetriever:
        def __init__(self, repository, index) -> None:
            calls["entity_retriever"] = {
                "repository": repository,
                "index": index,
            }

    class FakeChunkRetriever:
        def __init__(self, repository, index) -> None:
            calls["chunk_retriever"] = {
                "repository": repository,
                "index": index,
            }

    class FakeVectorFallbackGraphContextProvider:
        def __init__(self, graph_context_provider, entity_retriever) -> None:
            calls["graph_context_provider"] = {
                "graph_context_provider": graph_context_provider,
                "entity_retriever": entity_retriever,
            }

    class FakeSelector:
        def __init__(self) -> None:
            calls["selector"] = True

    class FakeOllamaOpenAIChatModel:
        @classmethod
        def from_settings(cls, settings):
            calls["model"] = {
                "base_url": settings.ollama_base_url,
                "model": settings.ollama_model,
                "think_mode": settings.ollama_think_mode,
                "timeout": settings.ollama_timeout,
            }
            return cls()

    class FakeWorkflow:
        def __init__(
            self,
            retriever,
            selector,
            model,
            retrieval_top_k: int,
            graph_context_provider=None,
            initial_entity_retriever=None,
            context_chunk_retriever=None,
            seed_paper_provider=None,
            query_understanding_prompt_template=None,
            agent_prompt_template=None,
            evaluation_prompt_template=None,
            retrieval_mode="graph",
        ) -> None:
            calls["workflow"] = {
                "retriever": retriever,
                "selector": selector,
                "model": model,
                "retrieval_top_k": retrieval_top_k,
                "graph_context_provider": graph_context_provider,
                "context_chunk_retriever": context_chunk_retriever,
                "seed_paper_provider": seed_paper_provider,
                "query_understanding_prompt_template": query_understanding_prompt_template,
                "agent_prompt_template": agent_prompt_template,
                "evaluation_prompt_template": evaluation_prompt_template,
                "retrieval_mode": retrieval_mode,
            }

    monkeypatch.setattr(bootstrap, "Neo4jGraphRepository", FakeRepository)
    monkeypatch.setattr(bootstrap, "OllamaEmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(bootstrap, "build_chroma_client", fake_build_chroma_client)
    monkeypatch.setattr(bootstrap, "ChromaVectorIndex", FakeChromaVectorIndex)
    monkeypatch.setattr(bootstrap, "HuggingFaceChunkTokenizer", FakeHuggingFaceChunkTokenizer)
    monkeypatch.setattr(bootstrap, "ChromaPaperRetriever", FakeRetriever)
    monkeypatch.setattr(bootstrap, "ChromaEntityRetriever", FakeEntityRetriever)
    monkeypatch.setattr(bootstrap, "ChromaChunkRetriever", FakeChunkRetriever)
    monkeypatch.setattr(bootstrap, "VectorFallbackGraphContextProvider", FakeVectorFallbackGraphContextProvider)
    monkeypatch.setattr(bootstrap, "TopScorePaperSelector", FakeSelector)
    monkeypatch.setattr(bootstrap, "OllamaOpenAIChatModel", FakeOllamaOpenAIChatModel)
    monkeypatch.setattr(bootstrap, "IdeaChatWorkflow", FakeWorkflow)

    settings = Settings(
        OPENALEX_BASE_URL="https://api.openalex.org",
        OLLAMA_BASE_URL="http://127.0.0.1:11434",
        OLLAMA_MODEL="qwen2.5:7b",
        OLLAMA_EMBED_MODEL="nomic-embed-text",
        OLLAMA_TIMEOUT=42.0,
        OLLAMA_THINK_MODE="low",
        AGENT_RETRIEVAL_TOP_K=7,
        PAPER_EMBEDDING_CACHE_DIR=str(tmp_path / "embeddings"),
        EMBEDDING_WORKER_COUNT=6,
        CHUNK_TOKENIZER_MODEL_PATH="/home/zb/models/Qwen3.6-35B-A3B-GPTQ-Int4",
        CHUNK_TOKENIZER_LOCAL_FILES_ONLY=True,
        CHUNK_MIN_TOKENS=256,
        CHUNK_MAX_TOKENS=2048,
        CHUNK_OVERLAP_TOKENS=80,
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="secret",
        NEO4J_DATABASE="idea",
    )

    workflow = bootstrap.build_idea_chat_workflow(settings)

    assert isinstance(workflow, FakeWorkflow)
    assert calls["repository"]["database"] == "idea"
    assert calls["embedder"]["model_name"] == "nomic-embed-text"
    assert calls["chroma_client_database"] == "idea"
    assert calls["vector_index"]["client"] == {"chroma": "client"}
    assert calls["vector_index"]["database_name"] == "idea"
    assert calls["chunk_tokenizer"] == {
        "model_path": "/home/zb/models/Qwen3.6-35B-A3B-GPTQ-Int4",
        "local_files_only": True,
    }
    assert calls["vector_index"]["tokenizer"] == {"tokenizer": "instance"}
    assert calls["vector_index"]["chunk_min_tokens"] == 256
    assert calls["vector_index"]["chunk_max_tokens"] == 2048
    assert calls["vector_index"]["chunk_overlap_tokens"] == 80
    assert calls["retriever"]["index"] is calls["vector_index_instance"]
    assert calls["entity_retriever"]["index"] is calls["vector_index_instance"]
    assert calls["chunk_retriever"]["index"] is calls["vector_index_instance"]
    assert calls["graph_context_provider"]["graph_context_provider"] == {"repository": "instance"}
    assert isinstance(calls["graph_context_provider"]["entity_retriever"], FakeEntityRetriever)
    assert calls["model"] == {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:7b",
        "think_mode": "low",
        "timeout": 42.0,
    }
    assert calls["workflow"]["retrieval_top_k"] == 7
    assert isinstance(calls["workflow"]["graph_context_provider"], FakeVectorFallbackGraphContextProvider)
    assert isinstance(calls["workflow"]["context_chunk_retriever"], FakeChunkRetriever)
    assert calls["workflow"]["retrieval_mode"] == "graph"
    assert calls["workflow"]["seed_paper_provider"] == {"repository": "instance"}
    assert "Extract search keywords" in calls["workflow"]["query_understanding_prompt_template"]
    assert "get_node_neighborhood" in calls["workflow"]["agent_prompt_template"]
    assert "Evaluation Agent" in calls["workflow"]["evaluation_prompt_template"]


def test_idea_chat_workflow_history_backtracking_and_validation(caplog):
    from src.workflows.idea_chat import IdeaChatWorkflow

    candidate = CandidatePaper(
        paper_id="Seed Paper",
        title="Seed Paper",
        abstract="Seed abstract",
        score=0.95,
    )
    
    neighborhoods = {
        "Seed Paper": {
            "entities": [{"id": "Seed Paper", "node_type": "Paper"}, {"id": "B", "node_type": "Concept"}, {"id": "E", "node_type": "Concept"}],
            "relationships": [{"source_id": "Seed Paper", "target_id": "B", "relationship_type": "CONTAINS"}, {"source_id": "Seed Paper", "target_id": "E", "relationship_type": "CONTAINS"}],
        },
        "B": {
            "entities": [{"id": "B", "node_type": "Concept"}, {"id": "C", "node_type": "Concept"}],
            "relationships": [{"source_id": "B", "target_id": "C", "relationship_type": "RELATED"}],
        },
        "C": {
            "entities": [{"id": "C", "node_type": "Concept"}, {"id": "D", "node_type": "Concept"}],
            "relationships": [{"source_id": "C", "target_id": "D", "relationship_type": "RELATED"}],
        },
        "D": {
            "entities": [{"id": "D", "node_type": "Concept"}],
            "relationships": [],
        },
        "X": {
            "entities": [{"id": "X", "node_type": "Concept"}],
            "relationships": [],
        }
    }

    class GraphContextProvider:
        def __init__(self):
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append(node_id)
            data = neighborhoods.get(node_id, {"entities": [], "relationships": []})
            return {
                "entities": data["entities"],
                "relationships": data["relationships"],
                "page": 1,
                "page_size": 20,
                "total_entities": len(data["entities"]),
                "total_relationships": len(data["relationships"]),
            }

    idea = IdeaOutput(
        title="Backtracked Idea",
        summary="Idea generated after backtracking.",
        problem="Graph path validation restricts agent steps.",
        method="Implement backtracking.",
        core_paper_id="Seed Paper",
        core_paper_title="Seed Paper",
    )

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            # Round 0: queries Seed Paper (valid)
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "Seed Paper"})],
            ),
            # Round 1: queries B (valid neighbor of Seed Paper)
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "B"})],
            ),
            # Round 2: queries C (valid neighbor of B)
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "C"})],
            ),
            # Round 3: backtracks to E (valid backtrack since E is in all_allowed_nodes and not yet queried)
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "E"})],
            ),
            # Round 4: queries X (invalid, not in cumulative history neighborhood B, C, D, E)
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "X"})],
            ),
            # Round 4 (retry): queries D (valid backtrack target)
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "D"})],
            ),
            # Round 5: stops
            FakeChatResult(content="Done"),
        ]
    )

    provider = GraphContextProvider()
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=provider,
    )

    with caplog.at_level(logging.INFO):
        result = workflow.run("graph idea generation")

    # Verify executed tool calls
    assert provider.calls == ["Seed Paper", "B", "C", "E", "X", "D"]
    assert result["idea_output"] == idea
    
    # Verify that the invalid call was warning-logged and retried
    assert "Round 4: Queried node 'X' not in history neighborhood. Retrying..." in caplog.text
    # Verify the user warning message was appended to guide backtracking
    user_msgs = [msg for msg in model.tool_calls[-1]["messages"] if msg.get("role") == "user"]
    warning_msg = user_msgs[-1]["content"]
    assert "节点 'X'" in warning_msg
    assert "不在已探索的图谱连通区域内" in warning_msg
    assert "你可以通过回溯（Backtracking）选择之前访问过的任何节点" in warning_msg


def test_idea_chat_workflow_react_width_enforcement(caplog):
    from src.workflows.idea_chat import IdeaChatWorkflow
    from src.agent.graph import settings

    # Explicitly set width to 3
    settings.graph_react_width = 3

    candidate = CandidatePaper(
        paper_id="Seed Paper",
        title="Seed Paper",
        abstract="Seed abstract",
        score=0.95,
    )
    
    neighborhoods = {
        "Seed Paper": {
            "entities": [
                {"id": "Seed Paper", "node_type": "Paper"},
                {"id": "B", "node_type": "Concept"},
                {"id": "C", "node_type": "Concept"},
                {"id": "D", "node_type": "Concept"},
                {"id": "E", "node_type": "Concept"}
            ],
            "relationships": [
                {"source_id": "Seed Paper", "target_id": "B", "relationship_type": "CONTAINS"},
                {"source_id": "Seed Paper", "target_id": "C", "relationship_type": "CONTAINS"},
                {"source_id": "Seed Paper", "target_id": "D", "relationship_type": "CONTAINS"},
                {"source_id": "Seed Paper", "target_id": "E", "relationship_type": "CONTAINS"}
            ],
        },
        "B": {
            "entities": [{"id": "B", "node_type": "Concept"}, {"id": "F", "node_type": "Concept"}],
            "relationships": [{"source_id": "B", "target_id": "F", "relationship_type": "RELATED"}],
        },
        "C": {
            "entities": [{"id": "C", "node_type": "Concept"}, {"id": "G", "node_type": "Concept"}],
            "relationships": [{"source_id": "C", "target_id": "G", "relationship_type": "RELATED"}],
        },
        "D": {
            "entities": [{"id": "D", "node_type": "Concept"}, {"id": "H", "node_type": "Concept"}],
            "relationships": [{"source_id": "D", "target_id": "H", "relationship_type": "RELATED"}],
        },
        "F": {"entities": [{"id": "F", "node_type": "Concept"}], "relationships": []},
        "G": {"entities": [{"id": "G", "node_type": "Concept"}], "relationships": []},
        "H": {"entities": [{"id": "H", "node_type": "Concept"}], "relationships": []},
    }

    class GraphContextProvider:
        def __init__(self):
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append(node_id)
            data = neighborhoods.get(node_id, {"entities": [], "relationships": []})
            return {
                "entities": data["entities"],
                "relationships": data["relationships"],
                "page": 1,
                "page_size": 20,
                "total_entities": len(data["entities"]),
                "total_relationships": len(data["relationships"]),
            }

    idea = IdeaOutput(
        title="Enforced Idea",
        summary="Idea generated.",
        problem="Test width limit.",
        method="Method description.",
        core_paper_id="Seed Paper",
        core_paper_title="Seed Paper",
    )

    model = FakeChatModel(
        structured_outputs=[
            {"keywords": ["graph"], "retrieval_query": "graph"},
            idea,
        ],
        tool_responses=[
            # Round 0: queries Seed Paper
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "Seed Paper"})],
            ),
            # Round 1: Agent tries to query only B (which is 1 node instead of 3). Invalid!
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "B"})],
            ),
            # Round 1 (retry): Agent corrects and queries B, C, D in parallel. Valid!
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "B"}),
                    FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "C"}),
                    FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "D"}),
                ],
            ),
            # Round 2: Agent queries F, G, H (which is 3 nodes). Valid!
            FakeChatResult(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "F"}),
                    FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "G"}),
                    FakeToolCall(tool_name="get_node_neighborhood", arguments={"node_id": "H"}),
                ],
            ),
            # Round 3: stops
            FakeChatResult(content="Done"),
        ]
    )

    provider = GraphContextProvider()
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever([candidate]),
        selector=StubSelector(),
        model=model,
        retrieval_top_k=3,
        graph_context_provider=provider,
    )

    with caplog.at_level(logging.INFO):
        result = workflow.run("graph idea generation")

    # Verify executed tool calls
    assert provider.calls == ["Seed Paper", "B", "B", "C", "D", "F", "G", "H"]
    assert result["idea_output"] == idea

    # Verify that the width violation retry warning was logged
    assert "Round 1: Agent did not call exactly 3 tools (got 1). Retrying..." in caplog.text
