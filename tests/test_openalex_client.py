import httpx
import logging

from src.ingestion.openalex_client import OpenAlexClient


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _status_error(url: str, status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("status error", request=request, response=response)


def test_search_works_requests_openalex_reference_filter(monkeypatch):
    captured = {}
    payload = {
        "results": [
            {"id": "https://openalex.org/W1", "title": "A", "referenced_works": ["https://openalex.org/W2"]},
            {"id": "https://openalex.org/W3", "title": "B", "referenced_works": []},
        ]
    }

    def fake_get(self, url, params):
        captured["url"] = url
        captured["params"] = params
        return DummyResponse(payload)

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    results = client.search_works("idea graph", must_have_references=True, per_page=10)

    assert captured["params"]["filter"] == "title_and_abstract.search:idea graph,referenced_works_count:>0"
    assert [item["id"] for item in results] == ["https://openalex.org/W1", "https://openalex.org/W3"]


def test_search_works_adds_openalex_api_key(monkeypatch):
    captured = {}

    def fake_get(self, url, params):
        captured["params"] = params.copy()
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org", api_keys=["key-a"])

    client.search_works("idea graph", must_have_references=False, per_page=10)

    assert captured["params"]["filter"] == "title_and_abstract.search:idea graph"
    assert captured["params"]["api_key"] == "key-a"


def test_search_works_logs_request_api_before_fetch(monkeypatch):
    messages = []

    def fake_get(self, url, params):
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org", log=messages.append)

    client.search_works(
        "idea graph",
        must_have_references=True,
        per_page=200,
        page=2,
        min_cited_by_count=0,
        min_publication_year=2024,
    )

    assert messages[0] == (
        "openalex:search_request url=https://api.openalex.org/works "
        "page=2 per_page=20 "
        "filter=title_and_abstract.search:idea graph,referenced_works_count:>0,"
        "cited_by_count:>0,from_publication_date:2024-01-01"
    )


def test_search_works_rotates_openalex_api_key_after_rate_limit(monkeypatch):
    captured = []

    def fake_get(self, url, params):
        captured.append(params.copy())
        if params.get("api_key") == "key-a":
            raise _status_error(url, 429)
        return DummyResponse({"results": [{"id": "https://openalex.org/W1"}]})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(
        base_url="https://api.openalex.org",
        api_keys=["key-a", "key-b"],
        api_key_cooldown_seconds=60,
    )

    results = client.search_works("idea graph", must_have_references=False, per_page=10)

    assert [item["id"] for item in results] == ["https://openalex.org/W1"]
    assert [params["api_key"] for params in captured] == ["key-a", "key-b"]


def test_search_skips_network_errors(monkeypatch):
    captured = {"messages": []}

    def fake_get(self, url, params):
        raise httpx.ReadTimeout("boom")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org", log=captured["messages"].append)

    assert client.search_works("idea graph", must_have_references=True, per_page=10) == []
    assert captured["messages"] == [
        "openalex:search_request url=https://api.openalex.org/works "
        "page=1 per_page=10 "
        "filter=title_and_abstract.search:idea graph,referenced_works_count:>0",
        "openalex:search_error keyword=idea graph error=ReadTimeout message=boom"
    ]


def test_get_work_uses_api_endpoint_for_openalex_url(monkeypatch):
    captured = {}

    def fake_get(self, url, params):
        captured["url"] = url
        return DummyResponse({"id": "https://openalex.org/W2"})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    result = client.get_work("https://openalex.org/W2")

    assert captured["url"] == "https://api.openalex.org/works/W2"
    assert result["id"] == "https://openalex.org/W2"


def test_get_work_logs_success_at_info_level(monkeypatch, caplog):
    def fake_get(self, url, params):
        return DummyResponse({"id": "https://openalex.org/W2"})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    with caplog.at_level(logging.INFO):
        client.get_work("https://openalex.org/W2")

    assert "openalex:get_work_done work=https://openalex.org/W2" in caplog.text


def test_search_all_works_combines_reference_and_citation_filters(monkeypatch):
    captured = []

    def fake_get(self, url, params):
        captured.append(params.copy())
        page = params["page"]
        if page == 1:
            return DummyResponse(
                {
                    "results": [
                        {"id": "https://openalex.org/W1", "title": "A", "referenced_works": ["https://openalex.org/W2"]},
                    ]
                }
            )
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    results = client.search_all_works(
        "idea graph",
        must_have_references=True,
        per_page=200,
        min_cited_by_count=50,
    )

    assert [item["id"] for item in results] == ["https://openalex.org/W1"]
    assert captured[0]["filter"] == "title_and_abstract.search:idea graph,referenced_works_count:>0,cited_by_count:>50"
    assert captured[0]["page"] == 1
    assert captured[1]["page"] == 2


def test_search_works_clamps_requested_per_page_to_twenty(monkeypatch):
    captured = {}

    def fake_get(self, url, params):
        captured["params"] = params.copy()
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    client.search_works("idea graph", must_have_references=False, per_page=200)

    assert captured["params"]["per-page"] == 20


def test_search_all_works_pages_through_multiple_openalex_pages(monkeypatch):
    captured = []

    def fake_get(self, url, params):
        captured.append(params.copy())
        page = params["page"]
        if page == 1:
            return DummyResponse({"results": [{"id": "https://openalex.org/W1"}] * 20})
        if page == 2:
            return DummyResponse({"results": [{"id": "https://openalex.org/W2"}] * 20})
        if page == 3:
            return DummyResponse({"results": [{"id": "https://openalex.org/W3"}] * 5})
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    results = client.search_all_works("idea graph", must_have_references=False, per_page=200)

    assert len(results) == 45
    assert [params["per-page"] for params in captured] == [20, 20, 20, 20]
    assert [params["page"] for params in captured] == [1, 2, 3, 4]


def test_search_works_adds_seed_publication_year_filter(monkeypatch):
    captured = {}

    def fake_get(self, url, params):
        captured["params"] = params.copy()
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    client.search_works(
        "idea graph",
        must_have_references=True,
        per_page=20,
        min_cited_by_count=50,
        min_publication_year=2021,
    )

    assert captured["params"]["filter"] == (
        "title_and_abstract.search:idea graph,referenced_works_count:>0,cited_by_count:>50,from_publication_date:2021-01-01"
    )


def test_search_works_appends_configured_openalex_filter(monkeypatch):
    captured = {}

    def fake_get(self, url, params):
        captured["params"] = params.copy()
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(
        base_url="https://api.openalex.org",
        works_filter="topics.field.id:17",
    )

    client.search_works(
        "GraphRAG retrieval augmented generation knowledge graph",
        must_have_references=True,
        per_page=20,
        min_publication_year=2021,
    )

    assert captured["params"]["filter"] == (
        "title_and_abstract.search:GraphRAG retrieval augmented generation knowledge graph,"
        "referenced_works_count:>0,from_publication_date:2021-01-01,topics.field.id:17"
    )


def test_search_works_includes_zero_cited_by_threshold(monkeypatch):
    captured = {}

    def fake_get(self, url, params):
        captured["params"] = params.copy()
        return DummyResponse({"results": []})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    client.search_works("idea graph", must_have_references=False, per_page=10, min_cited_by_count=0)

    assert captured["params"]["filter"] == "title_and_abstract.search:idea graph,cited_by_count:>0"
