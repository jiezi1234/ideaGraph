from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ChatIdeaTool:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[..., dict]

    def parameters_schema(self) -> dict[str, Any]:
        return self.args_schema.model_json_schema()

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    def call(self, arguments: dict[str, Any]) -> dict:
        parsed = self.args_schema.model_validate(arguments)
        return self.handler(**parsed.model_dump())
