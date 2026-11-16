import json

from src.schemas.agent import CanonicalIdea


def test_build_target_canonical_idea_uses_summary_motivation_and_design_details():
    from src.workflows.ai_idea_bench_i2i import build_target_canonical_idea

    record = {
        "target_paper": "Target Paper",
        "summary": {
            "motivation": "Target motivation.",
            "method": {
                "targeted_designs_details": [
                    {"design_name": "Design A", "description": "Use module A."},
                    {"design_name": "Design B", "description": "Evaluate module B."},
                ]
            },
        },
    }

    idea = build_target_canonical_idea(record)

    assert idea == CanonicalIdea(
        title="Target Paper",
        motivation="Target motivation.",
        experiment="Design A: Use module A.\nDesign B: Evaluate module B.",
    )


def test_load_generated_canonical_idea_prefers_archive_canonical_idea(tmp_path):
    from src.workflows.ai_idea_bench_i2i import load_generated_canonical_idea

    archive = tmp_path / "archive.json"
    archive.write_text(
        json.dumps(
            {
                "canonical_idea": {
                    "title": "Generated",
                    "motivation": "Generated motivation.",
                    "experiment": "Generated experiment.",
                },
                "idea_output": {"Title": "Raw title"},
            }
        ),
        encoding="utf-8",
    )

    idea = load_generated_canonical_idea(archive)

    assert idea == CanonicalIdea(
        title="Generated",
        motivation="Generated motivation.",
        experiment="Generated experiment.",
    )


def test_evaluate_idea_to_idea_uses_structured_judge_schema():
    from src.schemas.agent import AIIdeaBenchIdeaToIdeaEvaluation
    from src.workflows.ai_idea_bench_i2i import evaluate_idea_to_idea

    class FakeModel:
        def __init__(self):
            self.calls = []

        def structured_json(self, *, messages, schema, name: str, metadata=None):
            self.calls.append((messages, schema, name, metadata))
            return type(
                "Result",
                (),
                {
                    "parsed": schema(
                        motivation_similarity={"score": 4, "explanation": "Similar motivation."},
                        experiment_similarity={"score": 3, "explanation": "Partly similar experiment."},
                        overall_score=3.5,
                        summary="Close but not identical.",
                    )
                },
            )()

    model = FakeModel()
    generated = CanonicalIdea(
        title="Generated",
        motivation="Generated motivation.",
        experiment="Generated experiment.",
    )
    target = CanonicalIdea(
        title="Target",
        motivation="Target motivation.",
        experiment="Target experiment.",
    )

    result = evaluate_idea_to_idea(
        generated_idea=generated,
        target_idea=target,
        model=model,
    )

    assert isinstance(result, AIIdeaBenchIdeaToIdeaEvaluation)
    messages, schema, name, metadata = model.calls[0]
    assert schema is AIIdeaBenchIdeaToIdeaEvaluation
    assert name == "ai_idea_bench_i2i"
    assert metadata == {"stage": "ai_idea_bench_i2i"}
    assert "Generated motivation." in messages[1]["content"]
    assert "Target experiment." in messages[1]["content"]


def test_evaluate_idea_to_idea_official_uses_two_official_judges_without_titles():
    from src.schemas.agent import AIIdeaBenchOfficialIdeaToIdeaEvaluation
    from src.workflows.ai_idea_bench_i2i import evaluate_idea_to_idea_official

    class FakeModel:
        def __init__(self):
            self.calls = []

        def structured_json(self, *, messages, schema, name: str, metadata=None):
            self.calls.append((messages, schema, name, metadata))
            if name.endswith("motivation"):
                return type(
                    "Result",
                    (),
                    {
                        "parsed": {
                            "motivation_similarity": {
                                "rating": 4,
                                "explanation": "Motivations share the same problem focus.",
                            }
                        }
                    },
                )()
            return type(
                "Result",
                (),
                {
                    "parsed": {
                        "experiment_plan_similarity": {
                            "rating": 3,
                            "explanation": "Plans overlap in problem focus but differ in structure.",
                        }
                    }
                },
            )()

    model = FakeModel()
    generated = CanonicalIdea(
        title="Generated Title Should Not Be Sent",
        motivation="Generated motivation.",
        experiment="Generated experiment.",
    )
    target = CanonicalIdea(
        title="Target Title Should Not Be Sent",
        motivation="Target motivation.",
        experiment="Target experiment.",
    )

    result = evaluate_idea_to_idea_official(
        generated_idea=generated,
        target_idea=target,
        model=model,
    )

    assert isinstance(result, AIIdeaBenchOfficialIdeaToIdeaEvaluation)
    assert result.motivation_similarity.rating == 4
    assert result.experiment_plan_similarity.rating == 3
    assert len(model.calls) == 2
    assert model.calls[0][2] == "ai_idea_bench_official_i2i_motivation"
    assert model.calls[1][2] == "ai_idea_bench_official_i2i_experiment_plan"
    combined_user_content = "\n".join(call[0][1]["content"] for call in model.calls)
    assert "Generated motivation." in combined_user_content
    assert "Target experiment." in combined_user_content
    assert "Generated Title Should Not Be Sent" not in combined_user_content
    assert "Target Title Should Not Be Sent" not in combined_user_content
