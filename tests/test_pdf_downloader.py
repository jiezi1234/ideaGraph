
import httpx

from src.ingestion.pdf_downloader import PdfDownloader


class DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.request = httpx.Request("GET", "https://content.openalex.org/works/W1.pdf")

    def raise_for_status(self):
        if self.status_code >= 400:
            response = httpx.Response(self.status_code, request=self.request)
            raise httpx.HTTPStatusError("status error", request=self.request, response=response)


def test_pdf_downloader_adds_openalex_api_key_for_content_download(monkeypatch, tmp_path):
    captured = {}

    def fake_get(self, url, params=None):
        captured["url"] = url
        captured["params"] = params
        return DummyResponse(b"%PDF")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    downloader = PdfDownloader(openalex_api_keys=["key-a"])

    output = downloader.download(
        "https://content.openalex.org/works/W1.pdf",
        tmp_path / "paper.pdf",
    )

    assert output == tmp_path / "paper.pdf"
    assert output.read_bytes() == b"%PDF"
    assert captured["url"] == "https://content.openalex.org/works/W1.pdf"
    assert captured["params"] == {"api_key": "key-a"}


def test_pdf_downloader_rotates_openalex_api_key_after_rate_limit(monkeypatch, tmp_path):
    captured = []

    def fake_get(self, url, params=None):
        captured.append(params.copy())
        if params["api_key"] == "key-a":
            return DummyResponse(b"", status_code=429)
        return DummyResponse(b"%PDF")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    downloader = PdfDownloader(openalex_api_keys=["key-a", "key-b"])

    downloader.download("https://content.openalex.org/works/W1.pdf", tmp_path / "paper.pdf")

    assert [params["api_key"] for params in captured] == ["key-a", "key-b"]


def test_pdf_downloader_does_not_add_api_key_for_non_openalex_urls(monkeypatch, tmp_path):
    captured = {}

    def fake_get(self, url, params=None):
        captured["params"] = params
        return DummyResponse(b"%PDF")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    downloader = PdfDownloader(openalex_api_keys=["key-a"])

    downloader.download("https://repository.example.org/paper.pdf", tmp_path / "paper.pdf")

    assert captured["params"] is None
