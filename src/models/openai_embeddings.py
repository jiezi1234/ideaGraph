from __future__ import annotations

from openai import OpenAI

from src.models.openai_chat import normalize_base_url


class OpenAIEmbeddingModel:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 300.0,
    ) -> None:
        self._client = OpenAI(
            base_url=normalize_base_url(base_url),
            api_key=api_key or "unused",
            timeout=timeout,
        )
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model_name, input=text)
        return list(response.data[0].embedding)
