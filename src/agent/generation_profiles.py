from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from src.prompts import CHAT_IDEA_AGENT_PROMPT
from src.prompts import CHAT_IDEA_AI_IDEA_BENCH_PROMPT
from src.schemas.agent import AIIdeaBenchIdeaOutput
from src.schemas.agent import CanonicalIdea
from src.schemas.agent import IdeaOutput


@dataclass(frozen=True)
class GenerationProfile:
    # profile 同时约束提示词、结构化 schema 和评测归一化字段，避免只切 prompt 后解析失败。
    name: str
    prompt_variant: str
    prompt_template: str
    output_schema: type[BaseModel]
    internal_evaluation_enabled: bool
    canonicalizer: Callable[[BaseModel], CanonicalIdea]

    def canonicalize(self, idea_output: BaseModel) -> CanonicalIdea:
        return self.canonicalizer(idea_output)


def resolve_generation_profile(name: str | None) -> GenerationProfile:
    key = (name or "ideagraph").strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown generation profile: {name}. Allowed values: {allowed}") from exc


def _canonicalize_ideagraph(idea_output: BaseModel) -> CanonicalIdea:
    idea = idea_output if isinstance(idea_output, IdeaOutput) else IdeaOutput.model_validate(idea_output)
    experiment_parts = [idea.method.strip()]
    if idea.experiment.strip():
        experiment_parts.append(idea.experiment.strip())
    return CanonicalIdea(
        title=idea.title,
        motivation=idea.problem,
        experiment="\n\n".join(part for part in experiment_parts if part),
    )


def _canonicalize_ai_idea_bench(idea_output: BaseModel) -> CanonicalIdea:
    idea = (
        idea_output
        if isinstance(idea_output, AIIdeaBenchIdeaOutput)
        else AIIdeaBenchIdeaOutput.model_validate(idea_output)
    )
    return CanonicalIdea(
        title=idea.Title,
        motivation=idea.Motivation,
        experiment=idea.Experiment,
    )


_PROFILES = {
    "ideagraph": GenerationProfile(
        name="ideagraph",
        prompt_variant="ideagraph_v1",
        prompt_template=CHAT_IDEA_AGENT_PROMPT,
        output_schema=IdeaOutput,
        internal_evaluation_enabled=True,
        canonicalizer=_canonicalize_ideagraph,
    ),
    "ai_idea_bench": GenerationProfile(
        name="ai_idea_bench",
        prompt_variant="ai_idea_bench_v1",
        prompt_template=CHAT_IDEA_AI_IDEA_BENCH_PROMPT,
        output_schema=AIIdeaBenchIdeaOutput,
        internal_evaluation_enabled=False,
        canonicalizer=_canonicalize_ai_idea_bench,
    ),
}


GENERATION_PROFILE_CHOICES = tuple(sorted(_PROFILES))
