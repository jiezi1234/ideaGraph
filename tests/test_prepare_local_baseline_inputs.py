import json
import subprocess
from pathlib import Path

import scripts.prepare_local_baseline_inputs as prep
from scripts.prepare_local_baseline_inputs import (
    Neo4jConnectionConfig,
    PreparedPaper,
    build_ai_researcher_metaprompt,
    build_ai_researcher_instance,
    build_researchagent_context,
    extract_abstract_from_markdown,
    extract_reference_titles,
    fetch_paper_entities_from_neo4j,
    resolve_entities_for_context,
    write_baseline_inputs,
)
from scripts.run_researchagent_local_context import load_contexts, run_local_contexts


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_extracts_abstract_and_reference_titles_from_markdown():
    markdown = """# A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling

## Abstract

This paper proposes an evolutionary neural network approach for online bus scheduling.
It adapts dispatching decisions under uncertain passenger demand.

## 1 Introduction

Body text.

## References

[1] A Neural-Evolutionary Algorithm for Autonomous Transit Network Design. arXiv, 2024.
[2] Optimizing Electric Bus Charging Scheduling with Uncertainties Using Hierarchical Deep Reinforcement Learning. arXiv, 2025.
[3] Neural Networks for Vehicle Routing Problem. arXiv, 2024.
"""

    assert extract_abstract_from_markdown(markdown) == (
        "This paper proposes an evolutionary neural network approach for online bus scheduling. "
        "It adapts dispatching decisions under uncertain passenger demand."
    )
    assert extract_reference_titles(markdown, max_references=2) == [
        "A Neural-Evolutionary Algorithm for Autonomous Transit Network Design",
        "Optimizing Electric Bus Charging Scheduling with Uncertainties Using Hierarchical Deep Reinforcement Learning",
    ]


def test_extracts_ieee_inline_abstract_and_bulleted_references():
    markdown = """2024 International Conference metadata

# A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling

Authors

Abstract—Bus scheduling is vital to ensure service quality and save operational costs.
Current studies typically consider the problem as an optimization problem.

Index Terms—Public transport, bus scheduling, online scheduling.

I. INTRODUCTION

Body text.

REFERENCES

- [1] O. J. Ibarra-Rojas, F. Delgado, R. Giesen, and J. C. Muñoz, “Planning, operation, and control of bus transport systems: A literature review,” Transportation Research Part B: Methodological, vol. 77, pp. 38–75, 2015.
- [2] C. Wang, C. Guo, and X. Zuo, “Solving multi-depot electric vehicle scheduling problem by column generation and genetic algorithm,” Applied Soft Computing, vol. 112, p. 107774, 2021.
"""

    assert extract_abstract_from_markdown(markdown) == (
        "Bus scheduling is vital to ensure service quality and save operational costs. "
        "Current studies typically consider the problem as an optimization problem."
    )
    assert extract_reference_titles(markdown, max_references=2) == [
        "Planning, operation, and control of bus transport systems: A literature review",
        "Solving multi-depot electric vehicle scheduling problem by column generation and genetic algorithm",
    ]


def test_builds_ai_researcher_instance_from_local_and_cited_papers(tmp_path):
    target = PreparedPaper(
        title="A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling",
        abstract="Target abstract.",
        pdf_path=tmp_path / "target.pdf",
        markdown_path=tmp_path / "target.md",
        reference_titles=["Cited Reference A", "Cited Reference B"],
        full_text_context="Target full-text context with method and limitations.",
    )
    related = PreparedPaper(
        title="Related Local Scheduling Paper",
        abstract="Related abstract.",
        pdf_path=tmp_path / "related.pdf",
        markdown_path=tmp_path / "related.md",
        reference_titles=[],
        full_text_context="Related full-text context with baseline method.",
    )

    instance = build_ai_researcher_instance(
        target_paper=target,
        related_papers=[related],
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
        instance_id="bus_evo_nn",
        max_references=3,
    )

    assert instance["target"] == target.title
    assert instance["instance_id"] == "bus_evo_nn"
    assert instance["task2"] == "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"
    assert [paper["reference"] for paper in instance["source_papers"]] == [
        "Related Local Scheduling Paper",
        "Cited Reference A",
        "Cited Reference B",
    ]
    assert "Related full-text context with baseline method." in instance["source_papers"][0]["usage"]
    assert instance["source_papers"][1]["usage"] == "Use this cited paper as background evidence from the target paper bibliography."
    assert instance["local_paper_contexts"][0]["role"] == "target"
    assert instance["local_paper_contexts"][0]["context"] == "Target full-text context with method and limitations."


def test_builds_full_text_context_from_local_markdown_sections():
    markdown = """# Target Bus Scheduling Paper

Abstract—Short abstract about online bus scheduling.

I. INTRODUCTION

The introduction frames uncertain online bus dispatching as the core gap.

III. EVOLUTIONARY NEURAL NETWORK-BASED BUS SCHEDULING APPROACH

The method adapts NEAT to select buses at each departure time and convert bus duty types.

IV. EXPERIMENTAL RESULTS

Experiments use three Qingdao bus lines and compare ENNA with manual schedules.

V. CONCLUSION

Future research should validate the method in dynamic environments.

REFERENCES

- [1] This bibliography item should not enter the context.
"""

    context = prep.build_paper_full_text_context(
        title="Target Bus Scheduling Paper",
        abstract="Short abstract about online bus scheduling.",
        markdown=markdown,
        max_chars=1200,
    )

    assert "Abstract: Short abstract about online bus scheduling." in context
    assert "uncertain online bus dispatching" in context
    assert "adapts NEAT to select buses" in context
    assert "three Qingdao bus lines" in context
    assert "dynamic environments" in context
    assert "bibliography item should not enter" not in context


def test_full_text_context_does_not_treat_document_title_as_method_section():
    markdown = """# A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling

Authors and affiliations.

Abstract—The paper proposes ENNA.

I. INTRODUCTION

The introduction motivates the bus scheduling problem.

III. EVOLUTIONARY NEURAL NETWORK-BASED BUS SCHEDULING APPROACH

The actual method adapts NEAT to select buses at each departure time.

V. CONCLUSION

Future research should test dynamic environments.
"""

    context = prep.build_paper_full_text_context(
        title="A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling",
        abstract="The paper proposes ENNA.",
        markdown=markdown,
        max_chars=1200,
    )

    assert "Method details: The actual method adapts NEAT" in context
    assert "Method details: Authors and affiliations" not in context


def test_ai_researcher_metaprompt_includes_local_full_text_context(tmp_path):
    target = PreparedPaper(
        title="Target Bus Scheduling Paper",
        abstract="Target abstract.",
        pdf_path=tmp_path / "target.pdf",
        markdown_path=tmp_path / "target.md",
        reference_titles=[],
        full_text_context="Target full-text method context with NEAT scheduling.",
    )
    related = PreparedPaper(
        title="Related Local Paper",
        abstract="Related abstract.",
        pdf_path=tmp_path / "related.pdf",
        markdown_path=tmp_path / "related.md",
        reference_titles=[],
        full_text_context="Related full-text context about robust dispatching.",
    )

    contexts = prep.build_ai_researcher_paper_contexts(target, [related])
    metaprompt = build_ai_researcher_metaprompt(
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
        target_paper=target,
        related_papers=[related],
        entities=["bus vehicle scheduling"],
    )

    assert contexts == [
        {
            "role": "target",
            "title": "Target Bus Scheduling Paper",
            "abstract": "Target abstract.",
            "local_pdf": str(tmp_path / "target.pdf"),
            "local_markdown": str(tmp_path / "target.md"),
            "context": "Target full-text method context with NEAT scheduling.",
        },
        {
            "role": "related",
            "title": "Related Local Paper",
            "abstract": "Related abstract.",
            "local_pdf": str(tmp_path / "related.pdf"),
            "local_markdown": str(tmp_path / "related.md"),
            "context": "Related full-text context about robust dispatching.",
        },
    ]
    assert "No code resources are required" in metaprompt
    assert "Target full-text method context with NEAT scheduling." in metaprompt
    assert "Related full-text context about robust dispatching." in metaprompt


def test_writes_ai_researcher_and_researchagent_artifacts(tmp_path):
    target = PreparedPaper(
        title="Target Bus Scheduling Paper",
        abstract="Target abstract.",
        pdf_path=tmp_path / "target.pdf",
        markdown_path=tmp_path / "target.md",
        reference_titles=["Cited Reference A"],
    )
    related = PreparedPaper(
        title="Related Local Paper",
        abstract="Related abstract.",
        pdf_path=tmp_path / "related.pdf",
        markdown_path=tmp_path / "related.md",
        reference_titles=[],
    )

    summary = write_baseline_inputs(
        output_dir=tmp_path / "out",
        category="bus_sched",
        instance_id="bus_evo_nn",
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
        target_paper=target,
        related_papers=[related],
        entities=["bus vehicle scheduling", "evolutionary neural network"],
        max_references=2,
    )

    ai_json_path = tmp_path / "out/ai_researcher/benchmark/final/bus_sched/bus_evo_nn.json"
    metaprompt_path = tmp_path / "out/ai_researcher/benchmark/process/dataset_candidate/bus_sched/metaprompt.py"
    context_path = tmp_path / "out/researchagent/local_contexts.jsonl"
    manifest_path = tmp_path / "out/manifest.json"

    assert summary["ai_researcher_instance_path"] == str(ai_json_path)
    context_json_path = tmp_path / "out/ai_researcher/paper_contexts.json"
    ai_instance = json.loads(ai_json_path.read_text(encoding="utf-8"))
    assert ai_instance["source_papers"][0]["reference"] == "Related Local Paper"
    assert "TASK = r\"\"\"" in metaprompt_path.read_text(encoding="utf-8")
    assert "bus vehicle scheduling" in metaprompt_path.read_text(encoding="utf-8")
    assert summary["ai_researcher_context_path"] == str(context_json_path)
    assert json.loads(context_json_path.read_text(encoding="utf-8"))[0]["role"] == "target"

    context = json.loads(context_path.read_text(encoding="utf-8").strip())
    assert context == build_researchagent_context(
        target_paper=target,
        related_papers=[related],
        entities=["bus vehicle scheduling", "evolutionary neural network"],
        max_references=2,
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["target_paper"] == "Target Bus Scheduling Paper"
    commands = json.loads(manifest_path.read_text(encoding="utf-8"))["commands"]
    assert "run_ai_researcher_local_idea_agent.py" in commands["ai_researcher_local_idea_agent"]
    assert "--input-dir" in commands["ai_researcher_local_idea_agent"]
    assert "exports" not in commands["ai_researcher_local_idea_agent"]


def test_fetches_entities_from_neo4j_by_local_paper_id():
    calls = []

    class FakeDriver:
        def execute_query(self, query, *, parameters_, database_):
            calls.append({"query": query, "parameters": parameters_, "database": database_})
            return (
                [
                    {"entity_id": "evolutionary neural network", "entity_type": "Method"},
                    {"entity_id": "bus vehicle scheduling", "entity_type": "Problem"},
                    {"entity_id": "evolutionary neural network", "entity_type": "Method"},
                ],
                None,
                None,
            )

        def close(self):
            calls.append({"closed": True})

    def fake_driver_factory(uri, auth, **kwargs):
        calls.append({"uri": uri, "auth": auth, "kwargs": kwargs})
        return FakeDriver()

    entities = fetch_paper_entities_from_neo4j(
        "A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling",
        config=Neo4jConnectionConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="neo4j",
        ),
        driver_factory=fake_driver_factory,
    )

    assert entities == ["evolutionary neural network", "bus vehicle scheduling"]
    assert calls[0] == {
        "uri": "bolt://localhost:7687",
        "auth": ("neo4j", "password"),
        "kwargs": {"connection_timeout": 3.0, "max_transaction_retry_time": 0},
    }
    assert calls[1]["parameters"] == {
        "paper_id": "A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling"
    }
    assert calls[1]["database"] == "neo4j"
    assert "MATCH (:Paper {id: $paper_id})-[:CONTAINS]->(e:Entity)" in calls[1]["query"]
    assert calls[-1] == {"closed": True}


def test_resolves_entities_from_neo4j_before_manual_fallback(tmp_path):
    paper = PreparedPaper(
        paper_id="Local Paper Folder",
        title="Target Bus Scheduling Paper",
        abstract="Target abstract.",
        pdf_path=tmp_path / "target.pdf",
        markdown_path=tmp_path / "target.md",
        reference_titles=[],
    )

    assert resolve_entities_for_context(
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
        papers=[paper],
        explicit_entities=["manual fallback entity"],
        neo4j_entities=["graph entity", "graph entity"],
    ) == ["graph entity"]

    assert "manual fallback entity" in resolve_entities_for_context(
        question="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
        papers=[paper],
        explicit_entities=["manual fallback entity"],
        neo4j_entities=[],
    )


def test_researchagent_local_runner_writes_pipeline_results(tmp_path):
    context_path = tmp_path / "local_contexts.jsonl"
    result_path = tmp_path / "ideas.jsonl"
    context_path.write_text(
        json.dumps(
            {
                "paper": {"title": "Target", "abstract": "Target abstract."},
                "references": [],
                "entities": ["bus vehicle scheduling"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakePipeline:
        def run(self, context):
            context = dict(context)
            context["problem"] = "Generated problem"
            return context

    results = run_local_contexts(
        context_jsonl=context_path,
        results_path=result_path,
        pipeline_factory=lambda: FakePipeline(),
    )

    assert load_contexts(context_path)[0]["paper"]["title"] == "Target"
    assert results[0]["problem"] == "Generated problem"
    assert json.loads(result_path.read_text(encoding="utf-8").strip())["problem"] == "Generated problem"


def test_prepare_script_entrypoint_can_show_help_from_repo_root():
    result = subprocess.run(
        ["python3", "scripts/prepare_local_baseline_inputs.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Prepare local PDF papers" in result.stdout
