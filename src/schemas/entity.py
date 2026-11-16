from typing import Literal

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    entity_id: str
    entity_type: str
    description: str = ""
    source_refs: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    description_update_mode: Literal["append", "replace"] = "append"
    score: float | None = None


class ConceptResolutionResult(BaseModel):
    is_same: bool = Field(description="True if the two entities represent the same scientific concept, False otherwise")
    canonical_id: str | None = Field(default=None, description="The chosen canonical entity name (prefer standard full English name)")
    merged_description: str | None = Field(default=None, description="Merged description combining factual details from both concepts")

