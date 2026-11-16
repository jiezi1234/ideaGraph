from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.generation_profiles import resolve_generation_profile
from src.prompts.ai_idea_bench_official_i2i import EXPERIMENT_PLAN_SYSTEM_PROMPT
from src.prompts.ai_idea_bench_official_i2i import EXPERIMENT_PLAN_USER_PROMPT
from src.prompts.ai_idea_bench_official_i2i import MOTIVATION_SYSTEM_PROMPT
from src.prompts.ai_idea_bench_official_i2i import MOTIVATION_USER_PROMPT
from src.prompts.ai_idea_bench_i2i import PROMPT as AI_IDEA_BENCH_I2I_PROMPT
from src.schemas.agent import AIIdeaBenchIdeaToIdeaEvaluation
from src.schemas.agent import AIIdeaBenchOfficialExperimentPlanEvaluation
from src.schemas.agent import AIIdeaBenchOfficialIdeaToIdeaEvaluation
from src.schemas.agent import AIIdeaBenchOfficialMotivationEvaluation
from src.schemas.agent import CanonicalIdea


def build_target_canonical_idea(record: dict[str, Any]) -> CanonicalIdea:
    summary = record.get("summary") or {}
    method = summary.get("method") or {}
    details = method.get("targeted_designs_details") or []
    # 官方 I2I 比较的是 Motivation 和 Experiment，这里把 target 的设计细节压成实验侧 gold 文本。
    experiment_lines = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        name = str(detail.get("design_name") or "").strip()
        description = str(detail.get("description") or "").strip()
        if name and description:
            experiment_lines.append(f"{name}: {description}")
        elif description:
            experiment_lines.append(description)

    return CanonicalIdea(
        title=str(record.get("target_paper") or "").strip(),
        motivation=str(summary.get("motivation") or "").strip(),
        experiment="\n".join(experiment_lines),
    )


def load_generated_canonical_idea(path: str | Path) -> CanonicalIdea:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("canonical_idea"):
        return CanonicalIdea(**payload["canonical_idea"])

    idea_output = payload.get("idea_output")
    if not isinstance(idea_output, dict):
        raise ValueError(f"archive does not contain idea_output or canonical_idea: {path}")

    if {"Title", "Motivation", "Experiment"}.issubset(idea_output):
        profile = resolve_generation_profile("ai_idea_bench")
    else:
        profile = resolve_generation_profile("ideagraph")
    return profile.canonicalize(profile.output_schema(**idea_output))


def evaluate_idea_to_idea(
    *,
    generated_idea: CanonicalIdea,
    target_idea: CanonicalIdea,
    model,
    prompt_template: str = AI_IDEA_BENCH_I2I_PROMPT,
) -> AIIdeaBenchIdeaToIdeaEvaluation:
    content = (
        "<generated_idea>\n"
        f"Title: {generated_idea.title}\n"
        f"Motivation:\n{generated_idea.motivation}\n"
        f"Experiment:\n{generated_idea.experiment}\n"
        "</generated_idea>\n\n"
        "<target_idea>\n"
        f"Title: {target_idea.title}\n"
        f"Motivation:\n{target_idea.motivation}\n"
        f"Experiment:\n{target_idea.experiment}\n"
        "</target_idea>"
    )
    result = model.structured_json(
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": content},
        ],
        schema=AIIdeaBenchIdeaToIdeaEvaluation,
        name="ai_idea_bench_i2i",
        metadata={"stage": "ai_idea_bench_i2i"},
    )
    parsed = result.parsed
    return parsed if isinstance(parsed, AIIdeaBenchIdeaToIdeaEvaluation) else AIIdeaBenchIdeaToIdeaEvaluation(**parsed)


def evaluate_idea_to_idea_official(
    *,
    generated_idea: CanonicalIdea,
    target_idea: CanonicalIdea,
    model,
) -> AIIdeaBenchOfficialIdeaToIdeaEvaluation:
    # 官方 I2I 分别评 motivation 和 experiment plan；不要把 title 放进 judge 输入，
    # 否则 topic/title 词面重合会抬高相似度。
    motivation_result = model.structured_json(
        messages=[
            {"role": "system", "content": MOTIVATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MOTIVATION_USER_PROMPT.format(
                    motivation1=generated_idea.motivation,
                    motivation2=target_idea.motivation,
                ),
            },
        ],
        schema=AIIdeaBenchOfficialMotivationEvaluation,
        name="ai_idea_bench_official_i2i_motivation",
        metadata={"stage": "ai_idea_bench_official_i2i", "dimension": "motivation"},
    )
    experiment_result = model.structured_json(
        messages=[
            {"role": "system", "content": EXPERIMENT_PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXPERIMENT_PLAN_USER_PROMPT.format(
                    experiment_plan1=generated_idea.experiment,
                    experiment_plan2=target_idea.experiment,
                ),
            },
        ],
        schema=AIIdeaBenchOfficialExperimentPlanEvaluation,
        name="ai_idea_bench_official_i2i_experiment_plan",
        metadata={"stage": "ai_idea_bench_official_i2i", "dimension": "experiment_plan"},
    )
    motivation = motivation_result.parsed
    experiment = experiment_result.parsed
    if not isinstance(motivation, AIIdeaBenchOfficialMotivationEvaluation):
        motivation = AIIdeaBenchOfficialMotivationEvaluation(**motivation)
    if not isinstance(experiment, AIIdeaBenchOfficialExperimentPlanEvaluation):
        experiment = AIIdeaBenchOfficialExperimentPlanEvaluation(**experiment)
    return AIIdeaBenchOfficialIdeaToIdeaEvaluation(
        motivation_similarity=motivation.motivation_similarity,
        experiment_plan_similarity=experiment.experiment_plan_similarity,
    )
