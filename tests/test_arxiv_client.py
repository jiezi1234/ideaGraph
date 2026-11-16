from src.ingestion.arxiv_client import ArxivClient


class FakeHttpClient:
    def __init__(self, xml: str) -> None:
        self.xml = xml
        self.requests = []

    def get(self, url, params=None):
        self.requests.append((url, params))
        return FakeResponse(self.xml)


class FakeResponse:
    def __init__(self, xml: str) -> None:
        self.text = xml

    def raise_for_status(self) -> None:
        return None


def test_arxiv_client_finds_pdf_by_title():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Can LLMs Generate Novel Research Ideas?</title>
    <id>http://arxiv.org/abs/2409.04109v1</id>
    <link href="http://arxiv.org/abs/2409.04109v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2409.04109v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""
    fake = FakeHttpClient(xml)
    client = ArxivClient(http_client=fake)

    result = client.find_pdf_by_title("Can LLMs Generate Novel Research Ideas?")

    assert result == "http://arxiv.org/pdf/2409.04109v1"
    assert fake.requests[0][1]["search_query"] == 'ti:"Can LLMs Generate Novel Research Ideas?"'


def test_arxiv_client_returns_none_when_no_pdf_entry_matches():
    fake = FakeHttpClient('<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    client = ArxivClient(http_client=fake)

    assert client.find_pdf_by_title("Missing Paper") is None
