import json
from pathlib import Path

import pytest

from src.schemas.agent import CriterionRanking, IdeaComparison


def _comparison(overall_first: str) -> IdeaComparison:
    criterion = CriterionRanking(first=overall_first, explanation="匿名 idea 更完整。")
    return IdeaComparison(
        overall_first=overall_first,
        overall_explanation="匿名比较结论。",
        novelty=criterion,
        clarity=criterion,
        feasibility=criterion,
        relevance=criterion,
        expected_effectiveness=criterion,
        evidence_grounding=criterion,
    )


class FakeStructuredResult:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeJudgeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        return FakeStructuredResult(self.outputs.pop(0))


def test_normalize_method_idea_supports_ideagraph_archive_and_researchagent_jsonl(tmp_path):
    from scripts.method_idea_format import normalize_method_idea_file

    ideagraph_path = tmp_path / "ideagraph.json"
    ideagraph_path.write_text(
        json.dumps(
            {
                "idea_output": {
                    "title": "Adaptive Depot-Aware Evolution",
                    "summary": "Generate bus scheduling ideas.",
                    "problem": "Depot constraints are ignored.",
                    "method": "Use graph-conditioned evolutionary neural search.",
                    "experiment": "Compare with static heuristics.",
                    "idea_lineage": "Built from graph neighborhoods.",
                    "core_paper_id": "paper-1",
                    "core_paper_title": "Bus scheduling",
                },
                "graph_context": {"entities": [{"id": "bus scheduling"}]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    researchagent_path = tmp_path / "researchagent.jsonl"
    researchagent_path.write_text(
        json.dumps(
            {
                "paper": {"title": "Bus scheduling"},
                "problem": "How can online bus scheduling adapt to uncertain demand?",
                "problem_rationale": "The core gap is online robustness.",
                "method": "Use a differentiable multi-objective neuro-evolution framework.",
                "method_feedbacks": {"Clarity": "Good"},
                "experiment": "Replay real bus logs and compare against dispatch baselines.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ideagraph = normalize_method_idea_file(
        source_path=ideagraph_path,
        method_name="IdeaGraph",
        source_format="auto",
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
    )
    researchagent = normalize_method_idea_file(
        source_path=researchagent_path,
        method_name="ResearchAgent",
        source_format="auto",
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
    )

    assert ideagraph.source_format == "ideagraph_archive"
    assert ideagraph.idea.method == "Use graph-conditioned evolutionary neural search."
    assert "graph-conditioned evolutionary neural search" in ideagraph.judge_text
    assert "graph_context" not in ideagraph.judge_text
    assert researchagent.source_format == "researchagent_jsonl"
    assert researchagent.idea.problem == "How can online bus scheduling adapt to uncertain demand?"
    assert "online bus scheduling adapt to uncertain demand" in researchagent.judge_text
    assert "The core gap is online robustness" not in researchagent.judge_text
    assert "method_feedbacks" not in researchagent.judge_text


def test_normalize_method_idea_supports_ai_researcher_local_idea_agent(tmp_path):
    from scripts.method_idea_format import normalize_method_idea_file

    ai_researcher_path = tmp_path / "ai_researcher_ideas.json"
    selected_candidate = (
        "# Research Analysis & Innovation Ideas\n\n"
        "## Idea 1: Multi-Objective Evolutionary Graph Neural Network\n\n"
        "This original candidate idea should be judged directly."
    )
    ai_researcher_path.write_text(
        json.dumps(
            {
                "category": "bus_sched",
                "instance_id": "bus_evo_nn",
                "initial_query": "Hidden prompt with method-specific runtime details.",
                "selected_idea": "\n".join(
                    [
                        "# Selected Innovative Idea Report: MO-EGNN-BSP",
                        "",
                        "## 1. Selection Rationale",
                        "Idea 1 is selected as the most promising direction.",
                        "",
                        "## 2. Challenges & Existing Methods Analysis",
                        "Static bus schedules fail under traffic and demand uncertainty.",
                        "",
                        "## 3. Motivation",
                        "Bus scheduling should be formulated as a dynamic graph decision problem.",
                        "",
                        "## 4. Proposed Method: MO-EGNN-BSP",
                        "Use a multi-objective evolutionary graph neural network with NSGA-III and MAML.",
                        "",
                        "## 6. Expected Outcomes & Evaluation",
                        "Evaluate fleet size, passenger delay, energy cost, and robustness on replay logs.",
                    ]
                ),
                "ideas": [selected_candidate, "candidate idea 2"],
                "papers": [{"title": "A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    normalized = normalize_method_idea_file(
        source_path=ai_researcher_path,
        method_name="AI-Researcher",
        source_format="auto",
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
    )

    assert normalized.source_format == "ai_researcher_local_idea_agent"
    assert normalized.idea.original == selected_candidate
    assert normalized.judge_text == f"Original:\n{selected_candidate}"
    assert "Selection Rationale" not in normalized.judge_text
    assert normalized.idea.notes["candidate_idea_count"] == 2
    assert normalized.idea.notes["selected_idea_index"] == 1
    assert "initial_query" not in normalized.judge_text
    assert "AI-Researcher" not in normalized.judge_text


def test_ai_researcher_normalization_raises_when_selected_idea_index_is_missing(tmp_path):
    from scripts.method_idea_format import normalize_method_idea_file

    ai_researcher_path = tmp_path / "ai_researcher_ideas.json"
    ai_researcher_path.write_text(
        json.dumps(
            {
                "selected_idea": "# Selected Innovative Idea Report\n\nNo candidate index is present.",
                "ideas": ["candidate idea 1"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cannot determine selected AI-Researcher idea index"):
        normalize_method_idea_file(
            source_path=ai_researcher_path,
            method_name="AI-Researcher",
            source_format="auto",
        )


def test_ai_researcher_normalization_raises_when_selected_idea_index_is_out_of_range(tmp_path):
    from scripts.method_idea_format import normalize_method_idea_file

    ai_researcher_path = tmp_path / "ai_researcher_ideas.json"
    ai_researcher_path.write_text(
        json.dumps(
            {
                "selected_idea": "# Selected Innovative Idea Report\n\nIdea 3 is selected.",
                "ideas": ["candidate idea 1"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="out of range"):
        normalize_method_idea_file(
            source_path=ai_researcher_path,
            method_name="AI-Researcher",
            source_format="auto",
        )


def test_judge_text_uses_only_five_structured_fields_when_original_is_absent():
    from scripts.method_idea_format import NormalizedIdea, NormalizedMethodIdea

    idea = NormalizedIdea(
        title="T",
        summary="S",
        problem="P",
        method="M",
        experiment="E",
        core_paper_title="Core paper should not be judged",
        idea_lineage="Lineage should not be judged",
    )
    normalized = NormalizedMethodIdea(
        method_name="AnyMethod",
        source_path="source.json",
        source_format="test",
        idea=idea,
        judge_text="",
    )

    from scripts.method_idea_format import normalize_method_idea_payload

    payload = {
        "idea_output": {
            "title": idea.title,
            "summary": idea.summary,
            "problem": idea.problem,
            "method": idea.method,
            "experiment": idea.experiment,
            "core_paper_title": idea.core_paper_title,
            "idea_lineage": idea.idea_lineage,
        }
    }
    normalized = normalize_method_idea_payload(
        payload,
        method_name="AnyMethod",
        source_path="source.json",
        source_format="ideagraph_archive",
    )

    assert normalized.judge_text == "Title:\nT\n\nSummary:\nS\n\nProblem:\nP\n\nMethod:\nM\n\nExperiment:\nE"
    assert "Core paper should not be judged" not in normalized.judge_text
    assert "Lineage should not be judged" not in normalized.judge_text


def test_normalize_script_writes_common_json(tmp_path):
    from scripts.normalize_method_idea_outputs import normalize_to_file

    source_path = tmp_path / "ideagraph.json"
    output_path = tmp_path / "normalized.json"
    source_path.write_text(
        json.dumps(
            {
                "idea_output": {
                    "title": "Graph idea",
                    "summary": "Summary",
                    "problem": "Problem",
                    "method": "Method",
                    "experiment": "Experiment",
                    "core_paper_id": "paper-1",
                    "core_paper_title": "Paper",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    normalized = normalize_to_file(
        source_path=source_path,
        output_path=output_path,
        method_name="IdeaGraph",
        source_format="auto",
        question="Q",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "method_idea_v1"
    assert payload["method_name"] == "IdeaGraph"
    assert payload["judge_text"] == normalized.judge_text


def test_run_pairwise_blind_comparison_reverses_order_and_maps_winners_to_methods(tmp_path):
    from scripts.compare_normalized_method_ideas import run_pairwise_blind_comparison
    from scripts.method_idea_format import NormalizedIdea, NormalizedMethodIdea, write_normalized_method_idea

    ideagraph = NormalizedMethodIdea(
        method_name="IdeaGraph",
        source_path="ideagraph.json",
        source_format="ideagraph_archive",
        question="Q",
        idea=NormalizedIdea(title="Graph idea", method="use retrieved semantic graph evidence"),
        judge_text="Title: Graph idea\nMethod: use retrieved semantic graph evidence.",
        source_payload={},
    )
    researchagent = NormalizedMethodIdea(
        method_name="ResearchAgent",
        source_path="researchagent.jsonl",
        source_format="researchagent_jsonl",
        question="Q",
        idea=NormalizedIdea(problem="Agent idea", method="stage problem, method, and experiment"),
        judge_text="Problem: Agent idea\nMethod: stage problem, method, and experiment.",
        source_payload={},
    )
    ideagraph_path = tmp_path / "ideagraph.normalized.json"
    researchagent_path = tmp_path / "researchagent.normalized.json"
    write_normalized_method_idea(ideagraph, ideagraph_path)
    write_normalized_method_idea(researchagent, researchagent_path)
    model = FakeJudgeModel([_comparison("idea_a"), _comparison("idea_b")])

    report = run_pairwise_blind_comparison(
        idea_paths=[ideagraph_path, researchagent_path],
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
        model=model,
    )

    pair = report["pairs"][0]
    assert [run["order"] for run in pair["blind_runs"]] == ["forward", "reversed"]
    assert pair["blind_runs"][0]["idea_a_method"] == "IdeaGraph"
    assert pair["blind_runs"][1]["idea_a_method"] == "ResearchAgent"
    assert [run["overall_winner_method"] for run in pair["blind_runs"]] == ["IdeaGraph", "IdeaGraph"]
    assert pair["aggregate"]["overall_winner"] == "IdeaGraph"
    assert pair["aggregate"]["order_sensitivity"] == "consistent"
    assert report["global_summary"]["pair_count"] == 1
    assert [call[2] for call in model.calls] == [
        "blind_method_comparison_ideagraph_vs_researchagent_forward",
        "blind_method_comparison_ideagraph_vs_researchagent_reversed",
    ]
    for messages, schema, _name, metadata in model.calls:
        assert schema is IdeaComparison
        assert metadata["stage"] == "blind_method_comparison"
        combined = "\n".join(message["content"] for message in messages)
        assert "IdeaGraph" not in combined
        assert "ResearchAgent" not in combined


def test_write_blind_comparison_report_outputs_json_and_markdown(tmp_path):
    from scripts.compare_normalized_method_ideas import write_report

    report = {
        "question": "Q",
        "idea_files": ["ideagraph.normalized.json", "researchagent.normalized.json"],
        "pairs": [
            {
                "method_a": "IdeaGraph",
                "method_b": "ResearchAgent",
                "blind_runs": [
                    {
                        "order": "forward",
                        "idea_a_method": "IdeaGraph",
                        "idea_b_method": "ResearchAgent",
                        "overall_winner_method": "IdeaGraph",
                        "comparison": _comparison("idea_a").model_dump(),
                    }
                ],
                "aggregate": {
                    "overall_winner": "IdeaGraph",
                    "order_sensitivity": "consistent",
                    "overall_win_counts": {"IdeaGraph": 1, "ResearchAgent": 0, "tie": 0},
                    "criterion_win_counts": {"novelty": {"IdeaGraph": 1, "ResearchAgent": 0, "tie": 0}},
                    "summary": "双顺序盲评均倾向 IdeaGraph。",
                },
            }
        ],
        "global_summary": {"pair_count": 1},
    }

    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    write_report(report, output_json=output_json, output_md=output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))["pairs"][0]["aggregate"]["overall_winner"] == "IdeaGraph"
    markdown = output_md.read_text(encoding="utf-8")
    assert "双顺序盲评均倾向 IdeaGraph" in markdown
    assert "forward" in markdown
