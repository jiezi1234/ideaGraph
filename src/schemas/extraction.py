from pydantic import BaseModel, Field

from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    input_token_count: int | None = None
    output_token_count: int | None = None
