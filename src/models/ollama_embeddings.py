from __future__ import annotations

try:
    from ollama import Client
except ModuleNotFoundError:  # pragma: no cover - exercised via monkeypatch in tests
    Client = None


class OllamaEmbeddingModel:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        if Client is None:
            raise RuntimeError("ollama package is not installed")
        client = Client(host=self._base_url, timeout=self._timeout)
        body = client.embed(model=self._model_name, input=text)
        return list(body["embeddings"][0])
