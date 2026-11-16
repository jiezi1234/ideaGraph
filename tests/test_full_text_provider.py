from pathlib import Path

from src.ingestion.full_text import FullTextProvider
from src.schemas.document import PaperDocument


class FakeArxivClient:
    def __init__(self):
        self.queries = []

    def find_pdf_by_title(self, title):
        self.queries.append(title)
        return "https://arxiv.org/pdf/1234.5678"


class FakeDownloader:
    def __init__(self):
        self.urls = []

    def download(self, url, output_path):
        self.urls.append(url)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"%PDF")
        return output_path


class FakeExtractor:
    def extract(self, pdf_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / "paper.md"
        markdown_path.write_text("# Title\n\nBody\n\n# References\n\nRefs", encoding="utf-8")
        return type("Result", (), {"markdown_path": markdown_path, "mode": "hybrid"})()


class SelectiveDownloader:
    def __init__(self, payloads):
        self.payloads = payloads
        self.urls = []

    def download(self, url, output_path):
        self.urls.append(url)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.payloads[url])
        return output_path


class FailingFirstExtractor:
    def __init__(self):
        self.paths = []

    def extract(self, pdf_path, output_dir):
        self.paths.append(Path(pdf_path))
        if len(self.paths) == 1:
            raise RuntimeError("cannot parse this PDF")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / "paper.md"
        markdown_path.write_text("# Recovered\n\nBody", encoding="utf-8")
        return type("Result", (), {"markdown_path": markdown_path, "mode": "fast"})()


def test_full_text_provider_uses_openalex_pdf_url_first(tmp_path):
    provider = FullTextProvider(
        output_dir=tmp_path,
        arxiv_client=FakeArxivClient(),
        downloader=FakeDownloader(),
        extractor=FakeExtractor(),
    )
    doc = PaperDocument(
        document_id="Paper Title",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Abstract.",
        pdf_url="https://publisher.org/paper.pdf",
        ingestion_status="ready",
    )

    enriched = provider.enrich(doc)

    assert enriched.text_source == "full_text_md"
    assert enriched.full_text_markdown == "# Title\n\nBody"
    assert enriched.full_text_markdown_path.endswith(".md")
    assert provider.downloader.urls == ["https://publisher.org/paper.pdf"]
    assert provider.arxiv_client.queries == []


def test_full_text_provider_reuses_existing_markdown_without_downloading(tmp_path):
    paper_dir = tmp_path / "Cached_Paper"
    markdown_path = paper_dir / "attempt_2" / "paper.md"
    markdown_path.parent.mkdir(parents=True)
    markdown_path.write_text("# Cached\n\nBody\n\n# References\n\nRefs", encoding="utf-8")
    arxiv = FakeArxivClient()
    downloader = FakeDownloader()
    provider = FullTextProvider(
        output_dir=tmp_path,
        arxiv_client=arxiv,
        downloader=downloader,
        extractor=FakeExtractor(),
    )
    doc = PaperDocument(
        document_id="Cached Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Abstract.",
        pdf_urls=["https://repository.example.org/paper.pdf"],
        ingestion_status="ready",
    )

    enriched = provider.enrich(doc)

    assert enriched.text_source == "full_text_md"
    assert enriched.full_text_markdown == "# Cached\n\nBody"
    assert enriched.full_text_markdown_path == str(markdown_path)
    assert markdown_path.read_text(encoding="utf-8") == "# Cached\n\nBody"
    assert downloader.urls == []
    assert arxiv.queries == []


def test_full_text_provider_falls_back_to_arxiv_title_query(tmp_path):
    arxiv = FakeArxivClient()
    downloader = FakeDownloader()
    provider = FullTextProvider(
        output_dir=tmp_path,
        arxiv_client=arxiv,
        downloader=downloader,
        extractor=FakeExtractor(),
    )
    doc = PaperDocument(
        document_id="Paper Without Pdf URL",
        source_type="openalex",
        source_ref="https://openalex.org/W2",
        abstract="Abstract.",
        ingestion_status="ready",
    )

    enriched = provider.enrich(doc)

    assert enriched.text_source == "full_text_md"
    assert arxiv.queries == ["Paper Without Pdf URL"]
    assert downloader.urls == ["https://arxiv.org/pdf/1234.5678"]


def test_full_text_provider_tries_next_pdf_candidate_when_parsing_fails(tmp_path):
    downloader = SelectiveDownloader(
        {
            "https://repository.example.org/bad.pdf": b"%PDF bad",
            "https://content.openalex.org/works/W1.pdf": b"%PDF good",
        }
    )
    extractor = FailingFirstExtractor()
    provider = FullTextProvider(
        output_dir=tmp_path,
        arxiv_client=FakeArxivClient(),
        downloader=downloader,
        extractor=extractor,
    )
    doc = PaperDocument(
        document_id="Paper Title",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Abstract.",
        pdf_url="https://repository.example.org/bad.pdf",
        pdf_urls=[
            "https://repository.example.org/bad.pdf",
            "https://content.openalex.org/works/W1.pdf",
        ],
        ingestion_status="ready",
    )

    enriched = provider.enrich(doc)

    assert enriched.text_source == "full_text_md"
    assert enriched.pdf_url == "https://content.openalex.org/works/W1.pdf"
    assert enriched.full_text_markdown == "# Recovered\n\nBody"
    assert downloader.urls == [
        "https://repository.example.org/bad.pdf",
        "https://content.openalex.org/works/W1.pdf",
    ]
    assert provider.arxiv_client.queries == []


def test_full_text_provider_falls_back_to_arxiv_after_candidates_fail(tmp_path):
    arxiv = FakeArxivClient()
    downloader = SelectiveDownloader(
        {
            "https://repository.example.org/not-pdf": b"<html>not a pdf</html>",
            "https://content.openalex.org/works/W1.pdf": b'{"error":"API key required"}',
            "https://arxiv.org/pdf/1234.5678": b"%PDF arxiv",
        }
    )
    provider = FullTextProvider(
        output_dir=tmp_path,
        arxiv_client=arxiv,
        downloader=downloader,
        extractor=FakeExtractor(),
    )
    doc = PaperDocument(
        document_id="Paper Needing Arxiv",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Abstract.",
        pdf_urls=[
            "https://repository.example.org/not-pdf",
            "https://content.openalex.org/works/W1.pdf",
        ],
        ingestion_status="ready",
    )

    enriched = provider.enrich(doc)

    assert enriched.text_source == "full_text_md"
    assert enriched.pdf_url == "https://arxiv.org/pdf/1234.5678"
    assert arxiv.queries == ["Paper Needing Arxiv"]
    assert downloader.urls == [
        "https://repository.example.org/not-pdf",
        "https://content.openalex.org/works/W1.pdf",
        "https://arxiv.org/pdf/1234.5678",
    ]
