import json
import logging

import pytest

import main
from src.schemas.agent import EvaluationCriterion, IdeaEvaluation, IdeaOutput


def test_main_dispatches_build_graph(monkeypatch):
    called = {}

    def fake_run(args):
        called["keyword"] = args.keyword

    monkeypatch.setattr(main, "run_build_graph", fake_run)
    monkeypatch.setattr("sys.argv", ["main.py", "build-graph", "--keyword", "idea graph"])

    main.main()

    assert called["keyword"] == "idea graph"


def test_main_logs_unhandled_command_exception(monkeypatch, caplog):
    def fake_run(args):
        raise RuntimeError("workflow failed")

    monkeypatch.setattr(main, "run_chat_idea", fake_run)
    monkeypatch.setattr("sys.argv", ["main.py", "chat-idea", "--query", "graph ideas"])

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="workflow failed"):
        main.main()

    assert "command:failed command=chat-idea" in caplog.text
    assert "Traceback" in caplog.text


def test_main_dispatches_build_openalex_seed_graph(monkeypatch):
    called = {}

    def fake_run(args):
        called["openalex_ids"] = args.openalex_ids

    monkeypatch.setattr(main, "run_build_openalex_seed_graph", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "build-openalex-seed-graph",
            "--openalex-id",
            "W1",
            "--openalex-id",
            "https://openalex.org/W2",
        ],
    )

    main.main()

    assert called["openalex_ids"] == ["W1", "https://openalex.org/W2"]


def test_main_dispatches_preview_entity_extraction(monkeypatch):
    called = {}

    def fake_run(args):
        called["markdown_path"] = args.markdown_path
        called["output"] = args.output

    monkeypatch.setattr(main, "run_preview_entity_extraction", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "preview-entity-extraction",
            "--md",
            "paper.md",
            "--output",
            "json",
        ],
    )

    main.main()

    assert called == {"markdown_path": "paper.md", "output": "json"}


def test_run_build_openalex_seed_graph_routes_ids(monkeypatch, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"

    class FakeWorkflow:
        def run_openalex_seed_graph(self, openalex_ids):
            assert openalex_ids == ["W1", "W2"]
            return {"paper_count": 2, "entity_count": 0, "citation_count": 1, "entity_relation_count": 0}

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_mvp_workflow", lambda settings: FakeWorkflow())

    args = type("Args", (), {"openalex_ids": ["W1", "W2"]})()

    main.run_build_openalex_seed_graph(args)

    assert "OpenAlex seed graph done:" in capsys.readouterr().out


def test_run_benchmark_accepted_ideas_routes_args(monkeypatch, tmp_path, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"

    class FakeWorkflow:
        def run(self, venue, query, max_papers, output_dir, decision_scope):
            assert venue == "ICLR.cc/2026/Conference"
            assert query == "rag"
            assert max_papers == 3
            assert output_dir == str(tmp_path)
            assert decision_scope == "accepted"
            return {"paper_count": 1, "output_dir": str(tmp_path)}

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_accepted_paper_benchmark_workflow", lambda settings: FakeWorkflow())

    args = type(
        "Args",
        (),
        {
            "venue": "ICLR.cc/2026/Conference",
            "query": "rag",
            "max_papers": 3,
            "output_dir": str(tmp_path),
            "decision_scope": "accepted",
        },
    )()

    main.run_benchmark_accepted_ideas(args)

    assert "OpenReview paper benchmark done:" in capsys.readouterr().out


def test_run_prepare_ai_idea_bench_pilot_routes_args(monkeypatch, tmp_path, capsys):
    class FakeWorkflow:
        def run(self, dataset_path, count, output_dir):
            assert dataset_path == "dataset.json"
            assert count == 10
            assert output_dir == str(tmp_path)
            return {"target_count": 10, "output_dir": str(tmp_path)}

    monkeypatch.setattr(main, "AIIdeaBenchPilotWorkflow", lambda: FakeWorkflow())

    args = type(
        "Args",
        (),
        {
            "dataset_json": "dataset.json",
            "count": 10,
            "output_dir": str(tmp_path),
        },
    )()

    main.run_prepare_ai_idea_bench_pilot(args)

    assert "AI Idea Bench pilot prepared:" in capsys.readouterr().out


def test_run_preview_entity_extraction_prints_json(monkeypatch, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"

    class FakeWorkflow:
        def run(self, markdown_path):
            assert markdown_path == "paper.md"
            return {
                "full_text": {
                    "entities": [
                        {
                            "entity_id": "graph planning",
                            "entity_type": "Method",
                            "description": "A planning method.",
                        }
                    ],
                    "relations": [],
                },
                "incremental": {"entities": [], "relations": []},
            }

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_entity_extraction_preview_workflow", lambda settings: FakeWorkflow())

    args = type("Args", (), {"markdown_path": "paper.md", "output": "json"})()

    main.run_preview_entity_extraction(args)

    assert json.loads(capsys.readouterr().out) == {
        "full_text": {
            "entities": [
                {
                    "entity_id": "graph planning",
                    "entity_type": "Method",
                    "description": "A planning method.",
                }
            ],
            "relations": [],
        },
        "incremental": {"entities": [], "relations": []},
    }


def test_run_preview_entity_extraction_prints_table(monkeypatch, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"

    class FakeWorkflow:
        def run(self, markdown_path):
            assert markdown_path == "paper.md"
            return {
                "full_text": {
                    "entities": [
                        {
                            "entity_id": "graph planning",
                            "entity_type": "Method",
                            "description": "A planning method.",
                        }
                    ],
                    "relations": [],
                },
                "incremental": {"entities": [], "relations": []},
            }

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_entity_extraction_preview_workflow", lambda settings: FakeWorkflow())

    args = type("Args", (), {"markdown_path": "paper.md", "output": "table"})()

    main.run_preview_entity_extraction(args)

    output = capsys.readouterr().out
    assert output.startswith("Full Text\nEntities\n")
    assert "│ entity_id      │ entity_type │ description        │" in output
    assert "Incremental\nEntities\n" in output
    assert "Relations\n" in output


def test_run_chat_idea_prints_structured_idea_output(monkeypatch, tmp_path, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"
        neo4j_database = "testidea"
        llm_model_generation = "qwen3.6:35b"
        embed_model = "qwen3-embedding:4b"
        agent_retrieval_top_k = 3
        chat_idea_archive_dir = str(tmp_path)

    class FakeWorkflow:
        def run(self, user_query: str):
            assert user_query == "graph-based ideas"
            return {
                "retrieval_query": "knowledge graph scientific idea generation",
                "query_keywords": ["knowledge graph", "scientific idea generation"],
                "idea_output": IdeaOutput(
                    title="Graph Ideas",
                    summary="Use graph retrieval to generate ideas.",
                    problem="Low-resource domains lack enough papers.",
                    method="Retrieve related papers and combine graph signals.",
                    core_paper_id="W1",
                    core_paper_title="Core Paper",
                ),
                "idea_evaluation": IdeaEvaluation(
                    overall_score=4,
                    summary="方向清晰，但验证设计需要进一步具体化。",
                    novelty=EvaluationCriterion(
                        score=4,
                        comment="组合方式有一定新意。",
                        revision_suggestion="明确相对现有方法的新增贡献。",
                    ),
                    clarity=EvaluationCriterion(
                        score=3,
                        comment="问题可理解，但方法细节不足。",
                        revision_suggestion="补充输入、输出和关键模块。",
                    ),
                    feasibility=EvaluationCriterion(
                        score=3,
                        comment="原型可做，但成本不清楚。",
                        revision_suggestion="限定最小实验范围。",
                    ),
                    relevance=EvaluationCriterion(
                        score=5,
                        comment="贴合生成图谱 idea 的主题。",
                        revision_suggestion="保持当前问题边界。",
                    ),
                    expected_effectiveness=EvaluationCriterion(
                        score=3,
                        comment="机制合理，但缺少基线。",
                        revision_suggestion="加入检索增强 baseline。",
                    ),
                    evidence_grounding=EvaluationCriterion(
                        score=2,
                        comment="当前评估只依据生成 idea 本身。",
                        revision_suggestion="在后续人工复核中补充论文证据。",
                    ),
                    strengths=["问题方向明确"],
                    major_issues=["实验设计不够具体"],
                    revision_instruction="先补充最小可复现实验设计。",
                ),
                "idea_thinking": "先看检索结果，再结合图谱上下文生成方案。",
            }

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_idea_chat_workflow", lambda settings, **kwargs: FakeWorkflow())

    args = type("Args", (), {"query": "graph-based ideas", "retrieval_mode": "chunk_vector"})()

    main.run_chat_idea(args)

    output = capsys.readouterr().out
    assert "Query Keywords: knowledge graph, scientific idea generation" in output
    assert "Retrieval Query: knowledge graph scientific idea generation" in output
    assert "Title: Graph Ideas" in output
    assert "Summary:\nUse graph retrieval to generate ideas." in output
    assert "Problem:\nLow-resource domains lack enough papers." in output
    assert "Method:\nRetrieve related papers and combine graph signals." in output
    assert "--- Final Idea Generation Result ---" in output
    assert "--- Final Idea Evaluation ---" in output
    assert "Overall Score: 4/5" in output
    assert "Novelty: 4/5" in output
    assert "Clarity: 3/5" in output
    assert "Issue: 实验设计不够具体" in output
    assert "Revision Instruction:\n先补充最小可复现实验设计。" in output
    assert "Core Paper: Core Paper (W1)" in output
    assert "N/A" not in output

# This test focuses on ensuring that the CLI output properly unescapes newline characters in the structured idea output and evaluation, making it more readable for users.
def test_run_chat_idea_unescapes_newlines(monkeypatch, tmp_path, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"
        neo4j_database = "testidea"
        llm_model_generation = "qwen3.6:35b"
        embed_model = "qwen3-embedding:4b"
        agent_retrieval_top_k = 3
        chat_idea_archive_dir = str(tmp_path)

    class FakeWorkflow:
        def run(self, user_query: str):
            return {
                "user_query": user_query,
                "idea_output": IdeaOutput(
                    title="Graph Ideas",
                    summary="Line1\\nLine2",
                    problem="Problem line1\\nProblem line2",
                    method="Method line1\\nMethod line2",
                    experiment="Experiment line1\\nExperiment line2",
                    core_paper_id="W1",
                    core_paper_title="Core Paper",
                ),
                "idea_evaluation": IdeaEvaluation(
                    overall_score=4,
                    summary="Eval line1\\nEval line2",
                    novelty=EvaluationCriterion(score=4, comment="Comment line1\\nComment line2", revision_suggestion="Rev line1\\nRev line2"),
                    clarity=EvaluationCriterion(score=4, comment="", revision_suggestion=""),
                    feasibility=EvaluationCriterion(score=4, comment="", revision_suggestion=""),
                    relevance=EvaluationCriterion(score=4, comment="", revision_suggestion=""),
                    expected_effectiveness=EvaluationCriterion(score=4, comment="", revision_suggestion=""),
                    evidence_grounding=EvaluationCriterion(score=4, comment="", revision_suggestion=""),
                    strengths=["Strength line1\\nStrength line2"],
                    major_issues=["Issue line1\\nIssue line2"],
                    revision_instruction="Revision line1\\nRevision line2",
                ),
                "idea_thinking": "Think line1\\nThink line2",
            }

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_idea_chat_workflow", lambda settings, **kwargs: FakeWorkflow())

    args = type("Args", (), {"query": "graph-based ideas"})()

    main.run_chat_idea(args)

    output = capsys.readouterr().out
    assert "Summary:\nLine1\nLine2" in output
    assert "Problem:\nProblem line1\nProblem line2" in output
    assert "Method:\nMethod line1\nMethod line2" in output
    assert "Experiment:\nExperiment line1\nExperiment line2" in output
    assert "Judgment:\nEval line1\nEval line2" in output
    assert "Comment line1\nComment line2" in output
    assert "Revision line1\nRevision line2" in output
    assert "Strength: Strength line1\nStrength line2" in output
    assert "Issue: Issue line1\nIssue line2" in output


def test_run_chat_idea_writes_json_archive(monkeypatch, tmp_path, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"
        neo4j_database = "testidea"
        llm_model_generation = "qwen3.6:35b"
        embed_model = "qwen3-embedding:4b"
        agent_retrieval_top_k = 3
        chat_idea_archive_dir = str(tmp_path)

    class FakeWorkflow:
        def run(self, user_query: str):
            assert user_query == "graph-based ideas"
            return {
                "user_query": "graph-based ideas",
                "retrieval_query": "knowledge graph scientific idea generation",
                "query_keywords": ["knowledge graph", "scientific idea generation"],
                "candidate_papers": [
                    {
                        "paper_id": "W1",
                        "title": "Core Paper",
                        "abstract": "Graph retrieval improves ideation.",
                        "cited_by_count": 12,
                        "score": 0.91,
                    }
                ],
                "selected_paper": {
                    "paper_id": "W1",
                    "title": "Core Paper",
                    "abstract": "Graph retrieval improves ideation.",
                    "cited_by_count": 12,
                    "score": 0.91,
                },
                "graph_context": {"entities": [], "relationships": [], "page": 1, "page_size": 20},
                "tool_trace": [
                    {
                        "round": 1,
                        "tool_calls": [
                            {
                                "tool_name": "get_node_neighborhood",
                                "arguments": {"node_id": "Core Paper"},
                            }
                        ],
                        "observation": {"entities": [], "relationships": []},
                    }
                ],
                "idea_output": IdeaOutput(
                    title="Graph Ideas",
                    summary="Use graph retrieval to generate ideas.",
                    problem="Low-resource domains lack enough papers.",
                    method="Retrieve related papers and combine graph signals.",
                    experiment="Compare against retrieval-only baselines.",
                    core_paper_id="W1",
                    core_paper_title="Core Paper",
                    idea_lineage="这个想法由核心论文和图谱邻居中的方法关系共同启发。",
                ),
                "idea_evaluation": None,
                "idea_thinking": "先看检索结果。",
            }

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_idea_chat_workflow", lambda settings, **kwargs: FakeWorkflow())

    args = type("Args", (), {"query": "graph-based ideas", "retrieval_mode": "chunk_vector"})()

    main.run_chat_idea(args)

    archives = list(tmp_path.glob("*.json"))
    assert len(archives) == 1
    record = json.loads(archives[0].read_text(encoding="utf-8"))
    assert record["user_query"] == "graph-based ideas"
    assert record["retrieval_query"] == "knowledge graph scientific idea generation"
    assert record["candidate_papers"][0]["paper_id"] == "W1"
    assert record["selected_paper"]["paper_id"] == "W1"
    assert record["tool_trace"][0]["tool_calls"][0]["arguments"]["node_id"] == "Core Paper"
    assert record["idea_output"]["experiment"] == "Compare against retrieval-only baselines."
    assert record["idea_output"]["idea_lineage"] == "这个想法由核心论文和图谱邻居中的方法关系共同启发。"
    assert record["idea_thinking"] == "先看检索结果。"
    assert record["retrieval_mode"] == "chunk_vector"
    assert record["config"] == {
        "neo4j_database": "testidea",
        "llm_model_generation": "qwen3.6:35b",
        "embed_model": "qwen3-embedding:4b",
        "agent_retrieval_top_k": 3,
    }
    stdout = capsys.readouterr().out
    assert "Idea Lineage:" in stdout
    assert "这个想法由核心论文和图谱邻居中的方法关系共同启发。" in stdout
    assert "Archived Result:" in stdout


def test_run_chat_idea_routes_seed_paper_id(monkeypatch, tmp_path, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"
        neo4j_database = "testidea"
        llm_model_generation = "qwen3.6:35b"
        embed_model = "qwen3-embedding:4b"
        agent_retrieval_top_k = 3
        chat_idea_archive_dir = str(tmp_path)

    class FakeWorkflow:
        def run(self, user_query: str, seed_paper_id: str | None = None):
            assert user_query == "graph-based ideas"
            assert seed_paper_id == "Graph-guided discovery"
            return {
                "user_query": user_query,
                "seed_paper_id": seed_paper_id,
                "candidate_papers": [
                    {
                        "paper_id": "Graph-guided discovery",
                        "title": "Graph-guided discovery",
                        "abstract": "Graph retrieval improves ideation.",
                        "cited_by_count": 12,
                        "score": 1.0,
                    }
                ],
                "selected_paper": {
                    "paper_id": "Graph-guided discovery",
                    "title": "Graph-guided discovery",
                    "abstract": "Graph retrieval improves ideation.",
                    "cited_by_count": 12,
                    "score": 1.0,
                },
                "idea_output": IdeaOutput(
                    title="Seeded Graph Ideas",
                    summary="Use an explicit seed paper.",
                    problem="The paper is already known.",
                    method="Skip retrieval and generate from the selected paper.",
                    core_paper_id="Graph-guided discovery",
                    core_paper_title="Graph-guided discovery",
                ),
            }

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_idea_chat_workflow", lambda settings, **kwargs: FakeWorkflow())

    args = type(
        "Args",
        (),
        {
            "query": "graph-based ideas",
            "seed_paper_id": "Graph-guided discovery",
            "archive_output_dir": None,
        },
    )()

    main.run_chat_idea(args)

    archives = list(tmp_path.glob("*.json"))
    assert len(archives) == 1
    record = json.loads(archives[0].read_text(encoding="utf-8"))
    assert record["seed_paper_id"] == "Graph-guided discovery"
    assert record["selected_paper"]["paper_id"] == "Graph-guided discovery"
    assert "Core Paper: Graph-guided discovery (Graph-guided discovery)" in capsys.readouterr().out


def test_run_chat_idea_routes_generation_profile_and_retrieval_mode(monkeypatch, tmp_path, capsys):
    class FakeSettings:
        log_level = "INFO"
        log_file = "logs/test.log"
        neo4j_database = "testidea"
        llm_model_generation = "qwen3.6:35b"
        embed_model = "qwen3-embedding:4b"
        agent_retrieval_top_k = 3
        chat_idea_archive_dir = str(tmp_path)

    class FakeWorkflow:
        def run(self, user_query: str):
            assert user_query == "bench query"
            return {
                "idea_output": {
                    "Name": "bench_idea",
                    "Title": "Bench Idea",
                    "Motivation": "Motivation text.",
                    "Experiment": "Experiment text.",
                },
                "canonical_idea": {
                    "title": "Bench Idea",
                    "motivation": "Motivation text.",
                    "experiment": "Experiment text.",
                },
                "generation_profile": "ai_idea_bench",
                "prompt_variant": "ai_idea_bench_v1",
            }

    called = {}

    def fake_build_workflow(settings, generation_profile="ideagraph", retrieval_mode="graph"):
        del settings
        called["generation_profile"] = generation_profile
        called["retrieval_mode"] = retrieval_mode
        return FakeWorkflow()

    monkeypatch.setattr(main, "Settings", FakeSettings)
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_idea_chat_workflow", fake_build_workflow)

    args = type(
        "Args",
        (),
        {
            "query": "bench query",
            "generation_profile": "ai_idea_bench",
            "retrieval_mode": "chunk_vector",
            "seed_paper_id": None,
            "archive_output_dir": None,
        },
    )()

    main.run_chat_idea(args)

    assert called["generation_profile"] == "ai_idea_bench"
    assert called["retrieval_mode"] == "chunk_vector"
    output = capsys.readouterr().out
    assert "Title: Bench Idea" in output
    assert "Motivation:\nMotivation text." in output
    assert "Experiment:\nExperiment text." in output
