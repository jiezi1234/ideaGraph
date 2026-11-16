from __future__ import annotations

import httpx
import logging
import time


logger = logging.getLogger(__name__)
MAX_OPENALEX_PAGE_SIZE = 20


class OpenAlexClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 20.0,
        log=None,
        works_filter: str = "",
        api_keys: list[str] | None = None,
        api_key_cooldown_seconds: int = 300,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._info_log = log or logger.info
        self._error_log = log or logger.warning
        self._works_filter = works_filter.strip()
        self._api_keys = [key.strip() for key in api_keys or [] if key.strip()]
        self._api_key_cooldown_seconds = api_key_cooldown_seconds
        self._disabled_api_keys_until: dict[str, float] = {}
        self._api_key_index = 0

    def _normalize_per_page(self, per_page: int) -> int:
        return max(1, min(per_page, MAX_OPENALEX_PAGE_SIZE))

    def search_works(
        self,
        keyword: str,
        must_have_references: bool,
        per_page: int,
        page: int = 1,
        min_cited_by_count: int | None = None,
        min_publication_year: int | None = None,
    ) -> list[dict]:
        params = {
            "per-page": self._normalize_per_page(per_page),
            "page": page,
        }
        filters = [f"title_and_abstract.search:{keyword}"]
        if must_have_references:
            filters.append("referenced_works_count:>0")
        if min_cited_by_count is not None:
            filters.append(f"cited_by_count:>{min_cited_by_count}")
        if min_publication_year is not None:
            filters.append(f"from_publication_date:{min_publication_year}-01-01")
        if self._works_filter:
            filters.append(self._works_filter)
        if filters:
            params["filter"] = ",".join(filters)
        self._info_log(
            f"openalex:search_request url={self._base_url}/works "
            f"page={params['page']} per_page={params['per-page']} "
            f"filter={params.get('filter', '')}"
        )
        try:
            response = self._get(f"{self._base_url}/works", params)
        except httpx.HTTPError as exc:
            self._error_log(
                f"openalex:search_error keyword={keyword} "
                f"error={type(exc).__name__} message={exc}"
            )
            return []

        items = response.json().get("results", [])
        self._info_log(f"openalex:search_done keyword={keyword} results={len(items)}")
        return items

    def search_all_works(
        self,
        keyword: str,
        must_have_references: bool,
        per_page: int,
        min_cited_by_count: int | None = None,
        min_publication_year: int | None = None,
    ) -> list[dict]:
        items: list[dict] = []
        page = 1
        per_page = self._normalize_per_page(per_page)
        while True:
            batch = self.search_works(
                keyword,
                must_have_references=must_have_references,
                per_page=per_page,
                page=page,
                min_cited_by_count=min_cited_by_count,
                min_publication_year=min_publication_year,
            )
            if not batch:
                break
            items.extend(batch)
            page += 1
        return items

    def get_work(self, openalex_id: str) -> dict | None:
        work_url = self._build_work_url(openalex_id)
        try:
            response = self._get(work_url, {})
        except httpx.HTTPError as exc:
            self._error_log(
                f"openalex:get_work_error work={openalex_id} "
                f"error={type(exc).__name__} message={exc}"
            )
            return None

        self._info_log(f"openalex:get_work_done work={openalex_id}")
        return response.json()

    def _build_work_url(self, openalex_id: str) -> str:
        if openalex_id.startswith("http://") or openalex_id.startswith("https://"):
            work_id = openalex_id.rstrip("/").rsplit("/", 1)[-1]
            return f"{self._base_url}/works/{work_id}"
        return f"{self._base_url}/works/{openalex_id}"

    def _get(self, url: str, params: dict) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        attempts = max(1, len(self._api_keys))

        with httpx.Client(timeout=self._timeout) as client:
            for _ in range(attempts):
                request_params = params.copy()
                api_key = self._next_api_key()
                if api_key:
                    request_params["api_key"] = api_key
                try:
                    response = client.get(url, params=request_params)
                    response.raise_for_status()
                    return response
                except httpx.HTTPError as exc:
                    last_error = exc
                    if api_key and self._should_rotate_api_key(exc):
                        self._disable_api_key(api_key)
                        self._error_log(
                            f"openalex:api_key_disabled key_index={self._api_keys.index(api_key)} "
                            f"cooldown_seconds={self._api_key_cooldown_seconds} "
                            f"error={type(exc).__name__}"
                        )
                        continue
                    raise

        if last_error:
            raise last_error
        raise httpx.RequestError("OpenAlex request failed", request=httpx.Request("GET", url))

    def _next_api_key(self) -> str | None:
        if not self._api_keys:
            return None

        now = time.monotonic()
        for _ in range(len(self._api_keys)):
            key = self._api_keys[self._api_key_index]
            self._api_key_index = (self._api_key_index + 1) % len(self._api_keys)
            if self._disabled_api_keys_until.get(key, 0.0) <= now:
                return key
        return None

    def _disable_api_key(self, api_key: str) -> None:
        self._disabled_api_keys_until[api_key] = time.monotonic() + self._api_key_cooldown_seconds

    def _should_rotate_api_key(self, exc: httpx.HTTPError) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {401, 403, 429}
        return True
