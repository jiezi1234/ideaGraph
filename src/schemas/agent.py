import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CandidatePaper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    cited_by_count: int | None = None
    score: float = 0.0


class IdeaOutput(BaseModel):
    title: str
    summary: str
    problem: str
    method: str
    experiment: str = ""
    core_paper_id: str
    core_paper_title: str
    idea_lineage: str = ""

    @field_validator("problem", "method", "experiment", mode="before")
    @classmethod
    def normalize_section_text(cls, value):
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None)
        return value


class AIIdeaBenchIdeaOutput(BaseModel):
    Name: str
    Title: str
    Motivation: str
    Experiment: str
    Interestingness: int | None = None
    Feasibility: int | None = None
    Novelty: int | None = None


class CanonicalIdea(BaseModel):
    title: str
    motivation: str
    experiment: str


class SimilarityJudgment(BaseModel):
    score: int = Field(ge=1, le=5)
    explanation: str


class OfficialSimilarityJudgment(BaseModel):
    rating: int = Field(ge=1, le=5)
    explanation: str


class AIIdeaBenchIdeaToIdeaEvaluation(BaseModel):
    motivation_similarity: SimilarityJudgment
    experiment_similarity: SimilarityJudgment
    overall_score: float = Field(ge=1, le=5)
    summary: str


class AIIdeaBenchOfficialMotivationEvaluation(BaseModel):
    motivation_similarity: OfficialSimilarityJudgment


class AIIdeaBenchOfficialExperimentPlanEvaluation(BaseModel):
    experiment_plan_similarity: OfficialSimilarityJudgment


class AIIdeaBenchOfficialIdeaToIdeaEvaluation(BaseModel):
    motivation_similarity: OfficialSimilarityJudgment
    experiment_plan_similarity: OfficialSimilarityJudgment


class EvaluationCriterion(BaseModel):
    score: int
    comment: str
    revision_suggestion: str

    @model_validator(mode="before")
    @classmethod
    def normalize_criterion(cls, value):
        if isinstance(value, (int, float)):
            raise ValueError("Evaluation criterion cannot be a single score. Must have 'score', 'comment', and 'revision_suggestion'.")
        if isinstance(value, dict):
            # 容错：当模型输出中字段名发生轻微偏差时，进行映射修复。
            comment = value.get("comment")
            if comment is None:
                for alt in ["reason", "reasoning", "explanation", "analysis", "desc", "description"]:
                    if alt in value:
                        comment = value[alt]
                        break
            if comment is None:
                raise ValueError("Evaluation criterion is missing 'comment' (or synonyms like 'reason/explanation')")
            
            suggestion = value.get("revision_suggestion")
            if suggestion is None:
                for alt in ["suggestion", "suggest", "revision", "improvement", "advice"]:
                    if alt in value:
                        suggestion = value[alt]
                        break
            if suggestion is None:
                raise ValueError("Evaluation criterion is missing 'revision_suggestion' (or synonyms like 'suggestion/advice')")
                
            score = value.get("score")
            if score is None:
                for alt in ["rating", "value", "points", "grade"]:
                    if alt in value:
                        score = value[alt]
                        break
            if score is None:
                raise ValueError("Evaluation criterion is missing 'score' (or synonyms like 'rating/value')")
                
            return {
                "score": int(score),
                "comment": str(comment),
                "revision_suggestion": str(suggestion),
            }
        return value


class IdeaEvaluation(BaseModel):
    overall_score: int
    summary: str
    novelty: EvaluationCriterion
    clarity: EvaluationCriterion
    feasibility: EvaluationCriterion
    relevance: EvaluationCriterion
    expected_effectiveness: EvaluationCriterion
    evidence_grounding: EvaluationCriterion
    strengths: list[str]
    major_issues: list[str]
    revision_instruction: str

    @model_validator(mode="before")
    @classmethod
    def normalize_evaluation(cls, value):
        if not isinstance(value, dict):
            return value
            
        # 1. 容错 overall_score（允许以 overall, score 等替代）
        if "overall_score" not in value:
            for alt in ["overall", "score", "rating", "overall_rating"]:
                if alt in value and not isinstance(value[alt], dict):
                    value["overall_score"] = value[alt]
                    break
            if "overall_score" not in value:
                raise ValueError("IdeaEvaluation is missing 'overall_score' (or overall/score/rating)")
                
        # 2. 容错 summary（允许以 overview, comment 等替代）
        if "summary" not in value:
            for alt in ["summary_comment", "overview", "comment", "desc", "description"]:
                if alt in value:
                    value["summary"] = value[alt]
                    break
            if "summary" not in value:
                raise ValueError("IdeaEvaluation is missing 'summary' (or overview/comment/description)")
                
        # 3. 容错 strengths / major_issues，当这些列表缺失时自动用单数或相近词查找
        for list_field in ["strengths", "major_issues"]:
            if list_field not in value:
                for alt in [list_field[:-1], "advantage" if list_field == "strengths" else "weakness", "advantages" if list_field == "strengths" else "weaknesses", "issue" if list_field == "major_issues" else "strength", "issues" if list_field == "major_issues" else "strengths"]:
                    if alt in value:
                        value[list_field] = value[alt]
                        break
                if list_field not in value:
                    raise ValueError(f"IdeaEvaluation is missing '{list_field}'")
                    
        # 4. 容错 revision_instruction
        if "revision_instruction" not in value:
            for alt in ["revision_instructions", "instruction", "instructions", "suggested_revision", "revision_advice", "advice"]:
                if alt in value:
                    value["revision_instruction"] = value[alt]
                    break
            if "revision_instruction" not in value:
                raise ValueError("IdeaEvaluation is missing 'revision_instruction'")
                
        return value

    @field_validator("strengths", "major_issues", mode="before")
    @classmethod
    def normalize_list_field(cls, value):
        # Ollama 有时把 list[str] 字段输出为中文编号字符串（如 "1. xxx\n2. yyy"），
        # 在 Pydantic 层做容错转换，按换行或数字编号拆分成列表，避免校验崩溃。
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            items = re.split(r"\n+|\r\n+", value.strip())
            cleaned = []
            for item in items:
                item = re.sub(r"^\s*\d+[.、)]\s*", "", item).strip()
                if item:
                    cleaned.append(item)
            return cleaned or [value]
        return value


class CriterionRanking(BaseModel):
    first: Literal["idea_a", "idea_b", "tie"]
    explanation: str


class IdeaComparison(BaseModel):
    overall_first: Literal["idea_a", "idea_b", "tie"]
    overall_explanation: str
    novelty: CriterionRanking
    clarity: CriterionRanking
    feasibility: CriterionRanking
    relevance: CriterionRanking
    expected_effectiveness: CriterionRanking
    evidence_grounding: CriterionRanking
