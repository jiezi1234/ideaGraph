from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


class PaperAbstractExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abstract: str = Field(description="Concise abstract grounded in the provided paper text", min_length=1)


@dataclass(frozen=True)
class PaperAbstractExtractionResult:
    abstract: str
    input_token_count: int | None = None
    output_token_count: int | None = None


class PaperAbstractExtractor:
    def __init__(self, model, prompt_template: str, max_retries: int = 1) -> None:
        self._model = model
        self._prompt_template = prompt_template
        self._max_retries = max_retries

    def extract(self, paper_title: str, full_text_markdown: str) -> PaperAbstractExtractionResult:
        payload = {
            "title": paper_title,
            "content": full_text_markdown,
            "source_mode": "full_text_md",
        }
        retry_reason = None
        for attempt in range(self._max_retries + 1):
            prompt = self._build_prompt(retry_reason)
            try:
                parsed, usage = self._generate(prompt, payload)
                logger.info("abstract:raw_received title=%s keys=%s attempt=%s", paper_title, ["abstract"], attempt + 1)
                return PaperAbstractExtractionResult(
                    abstract=parsed.abstract.strip(),
                    input_token_count=usage.get("input"),
                    output_token_count=usage.get("output"),
                )
            except Exception as exc:
                retry_reason = str(exc)
                logger.warning(
                    "abstract:validation_failed title=%s attempt=%s reason=%s",
                    paper_title,
                    attempt + 1,
                    retry_reason,
                )
                if attempt >= self._max_retries:
                    raise
        raise RuntimeError("unreachable")

    def _generate(self, prompt: str, payload: dict) -> tuple[PaperAbstractExtractionOutput, dict]:
        result = self._model.structured_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": self._build_user_content(payload)},
            ],
            schema=PaperAbstractExtractionOutput,
            name="paper_abstract_extraction",
            metadata={"stage": "paper_abstract_extraction"},
        )
        parsed = result.parsed
        output = parsed if isinstance(parsed, PaperAbstractExtractionOutput) else PaperAbstractExtractionOutput.model_validate(parsed)
        return output, result.usage or {}

    def _build_user_content(self, payload: dict) -> str:
        return (
            f"Title: {payload.get('title', '')}\n"
            f"Source mode: {payload.get('source_mode', 'full_text_md')}\n\n"
            "<paper_markdown>\n"
            f"{payload.get('content', '')}\n"
            "</paper_markdown>"
        )

    def _build_prompt(self, retry_reason: str | None) -> str:
        if not retry_reason:
            return self._prompt_template
        return (
            f"{self._prompt_template}\n\n"
            "Previous output was invalid.\n"
            f"Reason: {retry_reason}\n"
            "Please re-extract and return content matching the provided schema exactly."
        )
