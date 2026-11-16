from typing import get_type_hints

import pytest

from src.schemas.agent import CandidatePaper, IdeaEvaluation, IdeaOutput


def test_query_understanding_node_trims_user_query():
    from src.agent.nodes.query_understanding import query_understanding

    result = query_understanding({"user_query": "  graph idea generation  "})

    assert result == {"retrieval_query": "graph idea generation"}


def test_query_understanding_node_extracts_keywords_with_model():
    from src.agent.nodes.query_understanding import QueryUnderstandingOutput, query_understanding

    class FakeStructuredResult:
        def __init__(self, parsed):
            self.parsed = parsed

    class FakeStructuredModel:
        def __init__(self):
            self.calls = []

        def structured_json(self, *, messages, schema, name: str, metadata=None):
            self.calls.append((messages, schema, name, metadata))
            return FakeStructuredResult(
                schema(
                    keywords=[
                        "knowledge graph",
                        "scientific idea generation",
                        "low-resource domains",
                    ],
                    retrieval_query="knowledge graph scientific idea generation low-resource domains",
                )
            )

    model = FakeStructuredModel()

    result = query_understanding(
        {"user_query": "  graph-based scientific idea generation for low-resource domains  "},
        model=model,
    )

    assert result == {
        "retrieval_query": "knowledge graph scientific idea generation low-resource domains",
        "query_keywords": [
            "knowledge graph",
            "scientific idea generation",
            "low-resource domains",
        ],
    }
    assert len(model.calls) == 1
    assert model.calls[0][1] is QueryUnderstandingOutput
    assert model.calls[0][2] == "query_understanding"
    assert model.calls[0][3] == {"stage": "query_understanding"}
    assert model.calls[0][0][0]["role"] == "system"
    assert model.calls[0][0][1]["role"] == "user"
    assert "graph-based scientific idea generation" in model.calls[0][0][1]["content"]


def test_query_understanding_node_falls_back_when_model_output_is_invalid():
    from src.agent.nodes.query_understanding import query_understanding

    class InvalidStructuredModel:
        def structured_json(self, *, messages, schema, name: str, metadata=None):
            return type("Result", (), {"parsed": {"keywords": "knowledge graph"}})()

    result = query_understanding(
        {"user_query": "  graph idea generation  "},
        model=InvalidStructuredModel(),
    )

    assert result == {
        "retrieval_query": "graph idea generation",
        "query_keywords": [],
    }


def test_query_understanding_node_requires_structured_json_model():
    from src.agent.nodes.query_understanding import query_understanding

    class JsonOnlyModel:
        pass

    result = query_understanding(
        {"user_query": "  graph idea generation  "},
        model=JsonOnlyModel(),
    )

    assert result == {
        "retrieval_query": "graph idea generation",
        "query_keywords": [],
    }


def test_agent_nodes_return_state_updates():
    from src.agent.nodes.paper_selection import paper_selection
    from src.agent.nodes.query_understanding import query_understanding
    from src.agent.state import AgentStateUpdate

    assert get_type_hints(query_understanding)["return"] is AgentStateUpdate
    assert get_type_hints(paper_selection)["return"] is AgentStateUpdate


def test_top_score_paper_selector_returns_top_scored_paper():
    from src.retrieval.selector import TopScorePaperSelector

    selector = TopScorePaperSelector()
    candidates = [
        CandidatePaper(
            paper_id="paper-1",
            title="Lower score",
            abstract="First abstract",
            score=0.7,
        ),
        CandidatePaper(
            paper_id="paper-2",
            title="Higher score",
            abstract="Second abstract",
            score=0.9,
        ),
    ]

    selected = selector.select("graph idea generation", candidates)

    assert selected.paper_id == "paper-2"


def test_paper_selection_node_uses_default_top_score_selector():
    from src.agent.nodes.paper_selection import paper_selection

    result = paper_selection(
        {
            "user_query": "graph idea generation",
            "candidate_papers": [
                CandidatePaper(
                    paper_id="paper-1",
                    title="Lower score",
                    abstract="First abstract",
                    score=0.5,
                ),
                CandidatePaper(
                    paper_id="paper-2",
                    title="Higher score",
                    abstract="Second abstract",
                    score=0.9,
                ),
            ],
        }
    )

    assert result["selected_paper"].paper_id == "paper-2"


def test_top_score_paper_selector_raises_on_empty_candidates():
    from src.retrieval.selector import TopScorePaperSelector

    selector = TopScorePaperSelector()

    with pytest.raises(ValueError, match="No candidate papers available for selection\\."):
        selector.select("graph idea generation", [])


def test_paper_selection_node_uses_selector():
    from src.agent.nodes.paper_selection import paper_selection

    selected = CandidatePaper(
        paper_id="paper-2",
        title="Higher score",
        abstract="Second abstract",
        score=0.9,
    )

    class StubSelector:
        def __init__(self):
            self.calls = []

        def select(self, user_query, candidates):
            self.calls.append((user_query, candidates))
            return selected

    selector = StubSelector()
    candidates = [
        CandidatePaper(
            paper_id="paper-1",
            title="Lower score",
            abstract="First abstract",
            score=0.7,
        ),
        selected,
    ]

    result = paper_selection(
        {
            "user_query": "graph idea generation",
            "candidate_papers": candidates,
        },
        selector=selector,
    )

    assert result == {"selected_paper": selected}
    assert selector.calls == [("graph idea generation", candidates)]


def test_idea_evaluation_node_returns_structured_judge_output():
    from src.agent.nodes.idea_evaluation import idea_evaluation

    class FakeStructuredResult:
        def __init__(self, parsed):
            self.parsed = parsed

    class FakeStructuredModel:
        def __init__(self):
            self.calls = []

        def structured_json(self, *, messages, schema, name: str, metadata=None):
            self.calls.append((messages, schema, name, metadata))
            return FakeStructuredResult(
                schema(
                    overall_score=4,
                    summary="方向有价值，但实验路径还需要收窄。",
                    novelty={
                        "score": 4,
                        "comment": "图谱上下文用于生成有一定新意。",
                        "revision_suggestion": "明确区别于普通 GraphRAG 的贡献。",
                    },
                    clarity={
                        "score": 3,
                        "comment": "问题和方法基本清楚。",
                        "revision_suggestion": "补充输入输出和关键模块。",
                    },
                    feasibility={
                        "score": 3,
                        "comment": "原型可做但评估成本不明。",
                        "revision_suggestion": "限定一个可验证数据集。",
                    },
                    relevance={
                        "score": 5,
                        "comment": "贴合用户查询。",
                        "revision_suggestion": "保持当前主题。",
                    },
                    expected_effectiveness={
                        "score": 3,
                        "comment": "可能有效但缺少基线。",
                        "revision_suggestion": "加入检索增强基线。",
                    },
                    evidence_grounding={
                        "score": 4,
                        "comment": "使用了核心论文和图谱上下文。",
                        "revision_suggestion": "补充更多相关论文对照。",
                    },
                    strengths=["有明确图谱依据"],
                    major_issues=["验证设计不够具体"],
                    revision_instruction="先补充可复现实验，再收窄方法贡献。",
                )
            )

    model = FakeStructuredModel()
    idea_output = IdeaOutput(
        title="图谱驱动的研究创意生成",
        summary="利用论文图谱上下文生成更有依据的研究创意。",
        problem="现有创意生成缺少可追溯的论文上下文。",
        method="结合候选论文摘要和一跳图谱邻域进行综合。",
        core_paper_id="paper-1",
        core_paper_title="Seed paper",
    )

    result = idea_evaluation(
        {
            "idea_output": idea_output,
        },
        model=model,
        prompt_template="<role>You are an Evaluation Agent.</role>",
    )

    assert result["idea_evaluation"].overall_score == 4
    assert result["idea_evaluation"].novelty.score == 4
    assert len(model.calls) == 1
    assert model.calls[0][1] is IdeaEvaluation
    assert model.calls[0][3] == {"stage": "evaluation"}
    assert "<role>You are an Evaluation Agent.</role>" in model.calls[0][0][0]["content"]
    user_content = model.calls[0][0][1]["content"]
    assert "图谱驱动的研究创意生成" in user_content
    assert "User query:" not in user_content
    assert "Core paper" not in user_content
    assert "Graph context:" not in user_content


def test_idea_evaluation_agent_can_call_graph_tool_before_scoring():
    from src.agent.nodes.idea_evaluation import idea_evaluation

    class FakeStructuredResult:
        def __init__(self, parsed):
            self.parsed = parsed

    class FakeToolCall:
        def __init__(self, tool_name: str, arguments: dict, tool_call_id: str = "eval_call_1"):
            self.tool_name = tool_name
            self.arguments = arguments
            self.tool_call_id = tool_call_id

    class FakeChatResult:
        def __init__(self, content: str = "", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.finish_reason = "stop"
            self.reasoning = ""

    class FakeEvaluationAgentModel:
        def __init__(self):
            self.tool_calls = []
            self.structured_calls = []

        def chat_with_tools(self, *, messages, tools, name: str, metadata=None):
            self.tool_calls.append(
                {"messages": messages, "tools": tools, "name": name, "metadata": metadata}
            )
            if len(self.tool_calls) == 1:
                return FakeChatResult(
                    tool_calls=[
                        FakeToolCall(
                            "get_node_neighborhood",
                            {"node_id": "Seed paper", "node_types": ["Paper"]},
                        )
                    ]
                )
            return FakeChatResult(content="已有足够图谱证据，可以进行评测。")

        def structured_json(self, *, messages, schema, name: str, metadata=None):
            self.structured_calls.append((messages, schema, name, metadata))
            return FakeStructuredResult(
                schema(
                    overall_score=4,
                    summary="评测结合了公交调度论文的图谱证据。",
                    novelty={"score": 4, "comment": "提出了跨线路调度扩展。", "revision_suggestion": "补充基线。"},
                    clarity={"score": 4, "comment": "结构清晰。", "revision_suggestion": "细化状态定义。"},
                    feasibility={"score": 3, "comment": "需要真实线路数据。", "revision_suggestion": "限定试点线路。"},
                    relevance={"score": 5, "comment": "贴合公交调度。", "revision_suggestion": "保持问题边界。"},
                    expected_effectiveness={"score": 4, "comment": "机制有合理依据。", "revision_suggestion": "补充消融。"},
                    evidence_grounding={"score": 4, "comment": "引用了图谱邻域证据。", "revision_suggestion": "补充更多论文。"},
                    strengths=["利用了图谱证据"],
                    major_issues=["实验协议仍需细化"],
                    revision_instruction="基于图谱证据补充基线和约束指标。",
                )
            )

    class FakeGraphContextProvider:
        def __init__(self):
            self.calls = []

        def get_node_neighborhood(self, node_id, node_types=None, relationship_types=None, page=1):
            self.calls.append(
                {
                    "node_id": node_id,
                    "node_types": node_types or [],
                    "relationship_types": relationship_types or [],
                    "page": page,
                }
            )
            return {
                "entities": [
                    {"id": node_id, "node_type": "Paper", "abstract": "Bus scheduling paper."},
                    {"id": "invalid action masking", "entity_type": "Method", "description": "Avoids infeasible vehicle choices."},
                ],
                "relationships": [
                    {"source_id": node_id, "target_id": "invalid action masking", "relationship_type": "CONTAINS"}
                ],
                "page": 1,
                "page_size": 20,
                "total_entities": 2,
                "total_relationships": 1,
            }

    model = FakeEvaluationAgentModel()
    provider = FakeGraphContextProvider()
    idea_output = IdeaOutput(
        title="跨线路公交在线调度",
        summary="结合强化学习和图谱证据生成调度策略。",
        problem="公交调度在突发拥堵下容易违反约束。",
        method="用图谱检索到的约束和方法设计评测指标。",
        core_paper_id="paper-1",
        core_paper_title="Seed paper",
    )

    result = idea_evaluation(
        {
            "user_query": "多线路公交在线调度",
            "idea_output": idea_output,
            "selected_paper": CandidatePaper(
                paper_id="paper-1",
                title="Seed paper",
                abstract="Bus scheduling paper.",
                score=1.0,
            ),
        },
        model=model,
        prompt_template="<role>You are an Evaluation Agent.</role>",
        graph_context_provider=provider,
    )

    assert provider.calls == [
        {"node_id": "Seed paper", "node_types": ["Paper"], "relationship_types": [], "page": 1}
    ]
    assert result["idea_evaluation"].overall_score == 4
    assert result["evaluation_graph_context"]["total_entities"] == 2
    assert result["evaluation_tool_trace"][0]["tool_calls"][0]["tool_name"] == "get_node_neighborhood"
    assert model.tool_calls[0]["tools"][0]["function"]["name"] == "get_node_neighborhood"
    assert model.tool_calls[0]["metadata"] == {"stage": "evaluation_agent_round", "round": 1}
    first_tool_messages = model.tool_calls[0]["messages"]
    assert first_tool_messages[0]["role"] == "system"
    assert "User query: 多线路公交在线调度" in first_tool_messages[0]["content"]
    assert "Selected paper:" in first_tool_messages[0]["content"]
    assert first_tool_messages[1]["role"] == "user"
    assert "跨线路公交在线调度" in first_tool_messages[1]["content"]
    assert "User query:" not in first_tool_messages[1]["content"]
    assert "Selected paper:" not in first_tool_messages[1]["content"]
    structured_messages = model.structured_calls[0][0]
    assert structured_messages[-1]["role"] != "user"
    assert "invalid action masking" in "\n".join(message.get("content", "") for message in structured_messages)
    assert model.structured_calls[0][2] == "idea_evaluation"


def test_idea_evaluation_node_requires_structured_json_model():
    from src.agent.nodes.idea_evaluation import idea_evaluation

    class JsonOnlyModel:
        pass

    idea_output = IdeaOutput(
        title="图谱驱动的研究创意生成",
        summary="利用论文图谱上下文生成更有依据的研究创意。",
        problem="现有创意生成缺少可追溯的论文上下文。",
        method="结合候选论文摘要和一跳图谱邻域进行综合。",
        core_paper_id="paper-1",
        core_paper_title="Seed paper",
    )

    with pytest.raises(AttributeError):
        idea_evaluation(
            {
                "idea_output": idea_output,
            },
            model=JsonOnlyModel(),
            prompt_template="<role>You are an Evaluation Agent.</role>",
        )


def test_idea_evaluation_robustness_normalization():
    from src.schemas.agent import IdeaEvaluation

    # 包含大量偏离结构（如整型字段、交错的命名、备用属性名）的原始模型输出 JSON 字典
    raw_payload = {
        "overall": 4,
        "summary": "想法很新颖，但实验部分细节不足。",
        "novelty": {
            "rating": 4,
            "reasoning": "新颖度极高。",
            "improvement": "无可改进。"
        },
        "clarity": {
            "score": 3,
            "reason": "结构基本清晰。",  # 使用了备用词 'reason'，应该被自动提取为 comment
            "suggestion": "细化方法部分。"  # 使用了备用词 'suggestion'，应该被自动提取为 revision_suggestion
        },
        "feasibility": {
            "rating": 5,  # 使用了备用词 'rating'
            "comment": "可行性很高。",
            "revision_suggestion": "保持原样。"
        },
        "relevance": {
            "score": 4,
            "comment": "关联性很强。",
            "revision_suggestion": "无"
        },
        "expected_effectiveness": {
            "score": 3,
            "comment": "可能具有一定有效性。",
            "revision_suggestion": "补充消融实验。"
        },
        "evidence_grounding": {
            "score": 4,
            "comment": "图谱背景证据充足。",
            "revision_suggestion": "补充额外论文。"
        },
        "strengths": "1. 方法新颖\n2. 可行性强",  # 字符串列表格式，应该被自动切分为 list[str]
        "major_issues": "1. 实验较少",  # 字符串列表格式，应该被自动切分为 list[str]
        "instruction": "先补充消融实验再行微调。"  # 使用了备用词 'instruction'
    }

    # 执行 Pydantic 校验与数据清洗标准化
    evaluation = IdeaEvaluation(**raw_payload)

    # 验证整体评分与备用字段提取
    assert evaluation.overall_score == 4
    assert evaluation.summary == "想法很新颖，但实验部分细节不足。"
    
    # 验证整型评分标准化为 Criterion 结构
    assert evaluation.novelty.score == 4
    assert evaluation.novelty.comment == "新颖度极高。"
    assert evaluation.novelty.revision_suggestion == "无可改进。"

    # 验证备用词 comment/revision_suggestion 提取
    assert evaluation.clarity.score == 3
    assert evaluation.clarity.comment == "结构基本清晰。"
    assert evaluation.clarity.revision_suggestion == "细化方法部分。"

    # 验证备用评分词 rating 提取
    assert evaluation.feasibility.score == 5

    # 验证 Strengths 列表转换
    assert evaluation.strengths == ["方法新颖", "可行性强"]
    assert evaluation.major_issues == ["实验较少"]

    # 验证备用修改建议字段提取
    assert evaluation.revision_instruction == "先补充消融实验再行微调。"
