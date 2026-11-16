from typing import Literal

from pydantic import BaseModel, Field


class ExtractedRelation(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str = ""
    source_refs: list[str] = Field(default_factory=list)
    description_update_mode: Literal["append", "replace"] = "append"
    score: float | None = None
    attributes: dict[str, str | int | float] = Field(default_factory=dict)


class RelationResolutionResult(BaseModel):
    merged_description: str = Field(description="Merged relation description grounded only in provided evidence")


class PaperCitationRelation(BaseModel):
    source_paper_id: str
    target_paper_id: str
    relation_type: str = "CITES"
