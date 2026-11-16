from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


class PdfDownloader:
    def __init__(
        self,
        timeout: float = 60.0,
        openalex_api_keys: list[str] | None = None,
        openalex_api_key_cooldown_seconds: int = 300,
    ) -> None:
        self._timeout = timeout
        self._openalex_api_keys = [key.strip() for key in openalex_api_keys or [] if key.strip()]
        self._openalex_api_key_cooldown_seconds = openalex_api_key_cooldown_seconds
        self._disabled_openalex_keys_until: dict[str, float] = {}
        self._openalex_api_key_index = 0

    def download(self, url: str, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            response = self._get(client, url)
        output_path.write_bytes(response.content)
        return output_path

    def _get(self, client: httpx.Client, url: str) -> httpx.Response:
        if not _is_openalex_content_url(url):
            response = client.get(url)
            response.raise_for_status()
            return response

        # OpenAlex content PDF 需要 api_key；外部仓库 URL 不携带项目密钥。
        attempts = max(1, len(self._openalex_api_keys))
        last_error: httpx.HTTPError | None = None
        for _ in range(attempts):
            api_key = self._next_openalex_api_key()
            params = {"api_key": api_key} if api_key else None
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if api_key and self._should_rotate_openalex_api_key(exc):
                    self._disable_openalex_api_key(api_key)
                    continue
                raise
        if last_error:
            raise last_error
        raise httpx.RequestError("OpenAlex content download failed", request=httpx.Request("GET", url))

    def _next_openalex_api_key(self) -> str | None:
        if not self._openalex_api_keys:
            return None

        now = time.monotonic()
        for _ in range(len(self._openalex_api_keys)):
            key = self._openalex_api_keys[self._openalex_api_key_index]
            self._openalex_api_key_index = (self._openalex_api_key_index + 1) % len(self._openalex_api_keys)
            if self._disabled_openalex_keys_until.get(key, 0.0) <= now:
                return key
        return None

    def _disable_openalex_api_key(self, api_key: str) -> None:
        self._disabled_openalex_keys_until[api_key] = time.monotonic() + self._openalex_api_key_cooldown_seconds

    def _should_rotate_openalex_api_key(self, exc: httpx.HTTPError) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {401, 403, 429}
        return True


def _is_openalex_content_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "content.openalex.org"
