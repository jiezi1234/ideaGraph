from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx


logger = logging.getLogger(__name__)

_ACCEPT_MARKERS = ("accept", "oral", "spotlight", "poster")
_REJECT_MARKERS = ("reject", "withdrawn", "desk rejected", "desk reject", "not accepted")
_QUERY_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9]+")
_NUMERIC_PREFIX_PATTERN = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class OpenReviewPaper:
    openreview_id: str
    title: str
    abstract: str
    status: str
    keywords: list[str] = field(default_factory=list)
    pdf_url: str | None = None
    metadata: dict = field(default_factory=dict)
    iclr_scores: list[float] = field(default_factory=list)


class OpenReviewClient:
    def __init__(
        self,
        base_url: str = "https://api2.openreview.net",
        timeout: float = 30.0,
        http_client=None,
        log=None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http_client = http_client
        self._log = log or logger

    def fetch_venue_papers(self, venue: str, limit: int | None = None) -> list[OpenReviewPaper]:
        notes = self._fetch_notes_for_venue(venue, limit)
        papers = [normalize_openreview_note(note, base_url=self._base_url) for note in notes]
        self._info("openreview:papers_fetched venue=%s count=%s", venue, len(papers))
        return papers

    def fetch_review_scores(self, forum_id: str) -> list[float]:
        try:
            response = self._get(f"{self._base_url}/notes", params={"forum": forum_id})
        except httpx.HTTPError as exc:
            self._warning(
                "openreview:reviews_failed forum=%s error=%s message=%s",
                forum_id,
                type(exc).__name__,
                exc,
            )
            return []

        return extract_review_scores(_response_notes(response))

    def _fetch_notes_for_venue(self, venue: str, limit: int | None) -> list[dict]:
        if limit == 0:
            return []

        venue_ids = self._resolve_venue_ids(venue)
        notes: list[dict] = []
        seen_note_ids: set[str] = set()
        offset = 0
        page_size = 100
        for venue_id in venue_ids:
            offset = 0
            while limit is None or len(notes) < limit:
                request_limit = page_size
                if limit is not None:
                    request_limit = min(page_size, limit - len(notes))
                params = {
                    "content.venueid": venue_id,
                    "limit": request_limit,
                    "offset": offset,
                }
                try:
                    response = self._get(f"{self._base_url}/notes", params=params)
                except httpx.HTTPError as exc:
                    self._warning(
                        "openreview:notes_failed venue=%s venue_id=%s offset=%s error=%s message=%s",
                        venue,
                        venue_id,
                        offset,
                        type(exc).__name__,
                        exc,
                    )
                    break

                batch = _response_notes(response)
                if not batch:
                    break
                for note in batch:
                    note_id = _as_text(note.get("id") or note.get("forum"))
                    if note_id and note_id in seen_note_ids:
                        continue
                    if note_id:
                        seen_note_ids.add(note_id)
                    notes.append(note)
                    if limit is not None and len(notes) >= limit:
                        break
                if len(batch) < request_limit or (limit is not None and len(notes) >= limit):
                    break
                offset += len(batch)

        return notes[:limit] if limit is not None else notes

    def _resolve_venue_ids(self, venue: str) -> list[str]:
        venue_ids = [venue]
        try:
            response = self._get(f"{self._base_url}/groups", params={"id": venue})
        except httpx.HTTPError as exc:
            self._warning(
                "openreview:venue_group_failed venue=%s error=%s message=%s",
                venue,
                type(exc).__name__,
                exc,
            )
            return venue_ids

        group = _response_groups(response)
        if not group:
            return venue_ids

        content = group[0].get("content") or {}
        for field_name in (
            "rejected_venue_id",
            "withdrawn_venue_id",
            "desk_rejected_venue_id",
        ):
            venue_id = _as_text(_content_field(content, field_name)).strip()
            if venue_id and venue_id not in venue_ids:
                venue_ids.append(venue_id)
        return venue_ids

    def _get(self, url: str, params: dict) -> httpx.Response:
        if self._http_client is not None:
            response = self._http_client.get(url, params=params)
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, params=params)
        response.raise_for_status()
        return response

    def _info(self, message: str, *args) -> None:
        if callable(self._log):
            self._log(message % args if args else message)
            return
        self._log.info(message, *args)

    def _warning(self, message: str, *args) -> None:
        if callable(self._log):
            self._log(message % args if args else message)
            return
        self._log.warning(message, *args)


def normalize_openreview_note(note: dict, base_url: str = "https://openreview.net") -> OpenReviewPaper:
    content = note.get("content") or {}
    title = _as_text(_content_field(content, "title"))
    abstract = _as_text(_content_field(content, "abstract"))
    status = _as_text(
        _first_present(
            _content_field(content, "decision"),
            _content_field(content, "venue"),
            _content_field(content, "venueid"),
        )
    )
    keywords = _as_text_list(_content_field(content, "keywords"))
    pdf_url = _normalize_pdf_url(
        _first_present(_content_field(content, "pdf"), note.get("pdf")),
        base_url=base_url,
    )
    openreview_id = _as_text(note.get("id") or note.get("forum"))

    return OpenReviewPaper(
        openreview_id=openreview_id,
        title=title,
        abstract=abstract,
        status=status,
        keywords=keywords,
        pdf_url=pdf_url,
        metadata=note.copy(),
    )


def filter_accepted_papers(papers: list[OpenReviewPaper]) -> list[OpenReviewPaper]:
    return [paper for paper in papers if _paper_is_accepted(paper)]


def filter_papers_by_decision_scope(
    papers: list[OpenReviewPaper],
    decision_scope: str,
) -> list[OpenReviewPaper]:
    scope = decision_scope.casefold().strip()
    if scope == "all":
        return papers
    if scope == "accepted":
        return [paper for paper in papers if _paper_decision_group(paper) == "accepted"]
    if scope == "rejected":
        return [paper for paper in papers if _paper_decision_group(paper) == "rejected"]
    if scope == "mixed":
        return [paper for paper in papers if _paper_decision_group(paper) in {"accepted", "rejected"}]
    raise ValueError(f"Unsupported decision scope: {decision_scope}")


def filter_papers_by_query(
    papers: list[OpenReviewPaper],
    query: str,
    *,
    include_metadata: bool = False,
) -> list[OpenReviewPaper]:
    tokens = _tokenize_text(query)
    if not tokens:
        return papers

    return [paper for paper in papers if _paper_matches_any_token(paper, tokens, include_metadata=include_metadata)]


def extract_review_scores(notes: list[dict]) -> list[float]:
    scores: list[float] = []
    for note in notes:
        content = note.get("content") or {}
        for field_name in ("rating", "recommendation", "score"):
            value = _content_field(content, field_name)
            for candidate in _as_iterable(value):
                score = _numeric_prefix(candidate)
                if score is not None:
                    scores.append(score)
                    break
            else:
                continue
            break
    return scores


def _response_notes(response: httpx.Response) -> list[dict]:
    payload = response.json()
    notes = payload.get("notes")
    return notes if isinstance(notes, list) else []


def _response_groups(response: httpx.Response) -> list[dict]:
    payload = response.json()
    groups = payload.get("groups")
    return groups if isinstance(groups, list) else []


def _paper_is_accepted(paper: OpenReviewPaper) -> bool:
    return _paper_decision_group(paper) == "accepted"


def _paper_decision_group(paper: OpenReviewPaper) -> str:
    status = paper.status.casefold()
    if any(marker in status for marker in _REJECT_MARKERS):
        return "rejected"
    if any(marker in status for marker in _ACCEPT_MARKERS):
        return "accepted"
    return "other"


def _content_field(content: dict, field_name: str):
    return _unwrap_value(content.get(field_name))


def _unwrap_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _first_present(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def _as_text(value) -> str:
    value = _unwrap_value(value)
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _as_text_list(value) -> list[str]:
    value = _unwrap_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def _normalize_pdf_url(value, base_url: str) -> str | None:
    pdf_url = _as_text(value).strip()
    if not pdf_url:
        return None
    return urljoin(base_url.rstrip("/") + "/", pdf_url)


def _paper_matches_any_token(
    paper: OpenReviewPaper,
    tokens: list[str],
    *,
    include_metadata: bool = False,
) -> bool:
    field_tokens = set(
        _tokenize_text(
            " ".join(
                [
                    paper.title,
                    paper.abstract,
                    " ".join(paper.keywords),
                    paper.status,
                ]
            )
        )
    )
    if include_metadata:
        field_tokens.update(_tokenize_text(_metadata_text(paper.metadata)))
    return any(token in field_tokens for token in tokens)


def _tokenize_text(text: str) -> list[str]:
    return [token.casefold() for token in _QUERY_TOKEN_PATTERN.split(text) if token]


def _metadata_text(metadata) -> str:
    return " ".join(_flatten_metadata_values(metadata))


def _flatten_metadata_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_metadata_values(item))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_metadata_values(item))
        return flattened
    text = str(value).strip()
    return [text] if text else []


def _numeric_prefix(value) -> float | None:
    match = _NUMERIC_PREFIX_PATTERN.match(str(value))
    if match is None:
        return None
    return float(match.group(1))


def _as_iterable(value):
    value = _unwrap_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
