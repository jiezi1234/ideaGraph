import asyncio
import time

from src.ingestion.openalex_service import OpenAlexPaperService


class FakeClient:
    def search_works(
        self,
        keyword,
        must_have_references,
        per_page,
        page=1,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        if page != 1:
            return []
        return [
            {
                "id": "https://openalex.org/W1",
                "title": "Seed Paper",
                "abstract_inverted_index": {"Seed": [0], "paper": [1]},
                "publication_year": 2024,
                "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
                "primary_location": {"pdf_url": "https://example.org/seed.pdf"},
            }
        ]

    def get_work(self, openalex_id):
        payloads = {
            "https://openalex.org/W2": {
                "id": "https://openalex.org/W2",
                "title": "Cited Paper B",
                "abstract_inverted_index": {"Cited": [0], "paper": [1]},
                "publication_year": 2023,
                "referenced_works": ["https://openalex.org/W3"],
            },
            "https://openalex.org/W3": {
                "id": "https://openalex.org/W3",
                "title": "Cited Paper C",
                "abstract_inverted_index": {"Another": [0], "paper": [1]},
                "publication_year": 2022,
                "referenced_works": [],
            },
        }
        return payloads[openalex_id]


def test_service_returns_seed_and_one_hop_cited_papers():
    service = OpenAlexPaperService(FakeClient())

    seed_docs, cited_docs, paper_links = service.fetch_seed_and_citations(
        keyword="idea graph",
        max_seed_papers=1,
        require_references=True,
    )

    assert [doc.document_id for doc in seed_docs] == ["Seed Paper"]
    assert seed_docs[0].source_ref == "https://openalex.org/W1"
    assert [doc.document_id for doc in cited_docs] == ["Cited Paper B", "Cited Paper C"]
    assert ("Seed Paper", "Cited Paper B") in {
        (link.source_paper_id, link.target_paper_id) for link in paper_links
    }
    assert ("Seed Paper", "Cited Paper C") in {
        (link.source_paper_id, link.target_paper_id) for link in paper_links
    }
    assert ("Cited Paper B", "Cited Paper C") in {
        (link.source_paper_id, link.target_paper_id) for link in paper_links
    }


def test_service_copies_openalex_pdf_url_to_document():
    service = OpenAlexPaperService(FakeClient())

    seed_docs, _ = service.fetch_seed_documents(
        keyword="idea graph",
        max_seed_papers=1,
        require_references=True,
    )

    assert seed_docs[0].pdf_url == "https://example.org/seed.pdf"


class OpenAlexPdfLocationClient:
    def __init__(self, raw_work):
        self.raw_work = raw_work

    def search_works(
        self,
        keyword,
        must_have_references,
        per_page,
        page=1,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        if page != 1:
            return []
        return [self.raw_work]


def test_service_orders_external_pdf_candidates_before_openalex_content_pdf():
    service = OpenAlexPaperService(
        OpenAlexPdfLocationClient(
            {
                "id": "https://openalex.org/W1569749180",
                "title": "Evolutionary Phenomena in Simple Dynamics",
                "abstract_inverted_index": {"Abstract": [0]},
                "referenced_works": [],
                "content_urls": {
                    "pdf": "https://content.openalex.org/works/W1569749180.pdf",
                },
                "best_oa_location": {
                    "pdf_url": "https://research.chalmers.se/en/publication/140676",
                    "landing_page_url": "https://research.chalmers.se/en/publication/140676",
                },
                "primary_location": {
                    "pdf_url": "https://publisher.example.org/article-page",
                },
                "open_access": {
                    "oa_url": "https://oa.example.org/article-page",
                },
            }
        )
    )

    seed_docs, _ = service.fetch_seed_documents(
        keyword="idea graph",
        max_seed_papers=1,
        require_references=False,
    )

    assert seed_docs[0].pdf_url == "https://research.chalmers.se/en/publication/140676"
    assert seed_docs[0].pdf_urls == [
        "https://research.chalmers.se/en/publication/140676",
        "https://publisher.example.org/article-page",
        "https://content.openalex.org/works/W1569749180.pdf",
    ]


def test_service_keeps_pdf_landing_page_in_openalex_candidates():
    service = OpenAlexPaperService(
        OpenAlexPdfLocationClient(
            {
                "id": "https://openalex.org/W1569749180",
                "title": "Evolutionary Phenomena in Simple Dynamics",
                "abstract_inverted_index": {"Abstract": [0]},
                "referenced_works": [],
                "best_oa_location": {
                    "pdf_url": "https://research.chalmers.se/en/publication/140676",
                    "landing_page_url": "https://research.chalmers.se/en/publication/140676",
                },
                "locations": [
                    {
                        "pdf_url": "https://repository.example.org/publication/140676",
                        "landing_page_url": "https://repository.example.org/publication/140676",
                    },
                    {
                        "pdf_url": None,
                        "landing_page_url": "http://publications.lib.chalmers.se/records/fulltext/140676/local_140676.pdf",
                    },
                ],
            }
        )
    )

    seed_docs, _ = service.fetch_seed_documents(
        keyword="idea graph",
        max_seed_papers=1,
        require_references=False,
    )

    assert (
        seed_docs[0].pdf_url
        == "https://research.chalmers.se/en/publication/140676"
    )
    assert seed_docs[0].pdf_urls == [
        "https://research.chalmers.se/en/publication/140676",
        "https://repository.example.org/publication/140676",
        "http://publications.lib.chalmers.se/records/fulltext/140676/local_140676.pdf",
    ]


class ManyRefsClient:
    def __init__(self):
        self.current = 0
        self.max_seen = 0

    def search_works(self, keyword, must_have_references, per_page, page=1, min_cited_by_count=None):
        if page != 1:
            return []
        refs = [f"https://openalex.org/W{i}" for i in range(2, 10)]
        return [
            {
                "id": "https://openalex.org/W1",
                "title": "Seed Paper",
                "abstract_inverted_index": {"Seed": [0], "paper": [1]},
                "publication_year": 2024,
                "referenced_works": refs[:per_page],
            }
        ]

    def get_work(self, openalex_id):
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        try:
            time.sleep(0.05)
            return {
                "id": openalex_id,
                "title": f"Paper {openalex_id.rsplit('/', 1)[-1]}",
                "abstract_inverted_index": {"Cited": [0], "paper": [1]},
                "publication_year": 2023,
                "referenced_works": [],
            }
        finally:
            self.current -= 1


def test_service_async_limits_openalex_reference_fetch_concurrency_to_five():
    client = ManyRefsClient()
    service = OpenAlexPaperService(client, max_concurrent_requests=5)

    seed_docs, cited_docs, paper_links = asyncio.run(
        service.fetch_seed_and_citations_async(
            keyword="idea graph",
            max_seed_papers=8,
            require_references=True,
        )
    )

    assert len(seed_docs) == 1
    assert len(cited_docs) == 8
    assert len(paper_links) == 8
    assert client.max_seen <= 5


class PriorityClient:
    def search_works(
        self,
        keyword,
        must_have_references,
        per_page,
        page=1,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        assert min_cited_by_count == 50
        assert min_publication_year == 2021
        if page == 1:
            return [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Seed A",
                    "abstract_inverted_index": {"Seed": [0], "A": [1]},
                    "publication_year": 2024,
                    "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
                    "cited_by_count": 80,
                }
            ]
        return []

    def get_work(self, openalex_id):
        payloads = {
            "https://openalex.org/W2": {
                "id": "https://openalex.org/W2",
                "title": "B",
                "abstract_inverted_index": {"B": [0]},
                "publication_year": 2023,
                "referenced_works": ["https://openalex.org/W4"],
                "cited_by_count": 120,
            },
            "https://openalex.org/W3": {
                "id": "https://openalex.org/W3",
                "title": "C",
                "abstract_inverted_index": {"C": [0]},
                "publication_year": 2023,
                "referenced_works": ["https://openalex.org/W5"],
                "cited_by_count": 20,
            },
            "https://openalex.org/W4": {
                "id": "https://openalex.org/W4",
                "title": "D",
                "abstract_inverted_index": {"D": [0]},
                "publication_year": 2022,
                "referenced_works": [],
                "cited_by_count": 10,
            },
            "https://openalex.org/W5": {
                "id": "https://openalex.org/W5",
                "title": "E",
                "abstract_inverted_index": {"E": [0]},
                "publication_year": 2022,
                "referenced_works": [],
                "cited_by_count": 5,
            },
        }
        return payloads[openalex_id]


def test_priority_seed_search_fetches_all_pages_with_citation_threshold():
    service = OpenAlexPaperService(PriorityClient())

    seed_docs, raw_seeds = asyncio.run(
        service.fetch_seed_documents_async(
            keyword="idea graph",
            max_seed_papers=1,
            require_references=True,
            fetch_all=True,
            min_cited_by_count=50,
            min_publication_year=2021,
        )
    )

    assert [doc.document_id for doc in seed_docs] == ["Seed A"]
    assert raw_seeds[0]["cited_by_count"] == 80


def test_fetch_referenced_documents_returns_raw_docs_and_direct_links():
    service = OpenAlexPaperService(PriorityClient())
    raw_paper = {
        "id": "https://openalex.org/W1",
        "title": "Seed A",
        "abstract_inverted_index": {"Seed": [0], "A": [1]},
        "publication_year": 2024,
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "cited_by_count": 80,
    }

    cited_docs, raw_cited, paper_links, failed_ids = asyncio.run(
        service.fetch_referenced_documents_async(raw_paper)
    )

    assert failed_ids == []
    assert [doc.document_id for doc in cited_docs] == ["B", "C"]
    assert [item["id"] for item in raw_cited] == ["https://openalex.org/W2", "https://openalex.org/W3"]
    assert {(link.source_paper_id, link.target_paper_id) for link in paper_links} == {
        ("Seed A", "B"),
        ("Seed A", "C"),
    }


class RaisingReferencedClient:
    def get_work(self, openalex_id):
        if openalex_id == "https://openalex.org/W3":
            raise RuntimeError("OpenAlex lookup failed")
        payloads = {
            "https://openalex.org/W2": {
                "id": "https://openalex.org/W2",
                "title": "B",
                "abstract_inverted_index": {"B": [0]},
                "publication_year": 2023,
                "referenced_works": [],
            },
            "https://openalex.org/W4": {
                "id": "https://openalex.org/W4",
                "title": "D",
                "abstract_inverted_index": {"D": [0]},
                "publication_year": 2022,
                "referenced_works": [],
            },
        }
        return payloads[openalex_id]


def test_fetch_referenced_documents_skips_raised_lookup_and_records_failed_id(caplog):
    service = OpenAlexPaperService(RaisingReferencedClient())
    raw_paper = {
        "id": "https://openalex.org/W1",
        "title": "Seed A",
        "abstract_inverted_index": {"Seed": [0], "A": [1]},
        "publication_year": 2024,
        "referenced_works": [
            "https://openalex.org/W2",
            "https://openalex.org/W3",
            "https://openalex.org/W4",
        ],
    }

    cited_docs, raw_cited, paper_links, failed_ids = service.fetch_referenced_documents(raw_paper)

    assert [doc.document_id for doc in cited_docs] == ["B", "D"]
    assert [item["id"] for item in raw_cited] == [
        "https://openalex.org/W2",
        "https://openalex.org/W4",
    ]
    assert failed_ids == ["https://openalex.org/W3"]
    assert {(link.source_paper_id, link.target_paper_id) for link in paper_links} == {
        ("Seed A", "B"),
        ("Seed A", "D"),
    }
    assert "openalex:get_work_skipped id=https://openalex.org/W3" in caplog.text


class OneHopRaisingClient(FakeClient):
    def get_work(self, openalex_id):
        if openalex_id == "https://openalex.org/W2":
            raise RuntimeError("OpenAlex lookup failed")
        return super().get_work(openalex_id)


def test_fetch_seed_and_citations_skips_raised_cited_lookup(caplog):
    service = OpenAlexPaperService(OneHopRaisingClient())

    seed_docs, cited_docs, paper_links = service.fetch_seed_and_citations(
        keyword="idea graph",
        max_seed_papers=1,
        require_references=True,
    )

    assert [doc.document_id for doc in seed_docs] == ["Seed Paper"]
    assert [doc.document_id for doc in cited_docs] == ["Cited Paper C"]
    assert {(link.source_paper_id, link.target_paper_id) for link in paper_links} == {
        ("Seed Paper", "Cited Paper C"),
    }
    assert "openalex:get_work_skipped id=https://openalex.org/W2" in caplog.text


class EmptyAbstractSeedClient:
    def search_works(
        self,
        keyword,
        must_have_references,
        per_page,
        page=1,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        if page != 1:
            return []
        return [
            {
                "id": "https://openalex.org/W1",
                "title": "Has Abstract",
                "abstract_inverted_index": {"Has": [0], "Abstract": [1]},
                "publication_year": 2024,
                "referenced_works": [],
            },
            {
                "id": "https://openalex.org/W2",
                "title": "Empty Abstract",
                "abstract_inverted_index": None,
                "publication_year": 2024,
                "referenced_works": [],
            },
        ]


def test_fetch_seed_documents_keeps_papers_without_abstract():
    service = OpenAlexPaperService(EmptyAbstractSeedClient())

    seed_docs, raw_seeds = service.fetch_seed_documents(
        keyword="idea graph",
        max_seed_papers=10,
        require_references=False,
    )

    assert [doc.document_id for doc in seed_docs] == ["Has Abstract", "Empty Abstract"]
    assert [doc.abstract for doc in seed_docs] == ["Has Abstract", ""]
    assert [item["id"] for item in raw_seeds] == ["https://openalex.org/W1", "https://openalex.org/W2"]


class EmptyAbstractReferencedClient:
    def get_work(self, openalex_id):
        payloads = {
            "https://openalex.org/W2": {
                "id": "https://openalex.org/W2",
                "title": "Has Abstract",
                "abstract_inverted_index": {"Has": [0], "Abstract": [1]},
                "publication_year": 2023,
                "referenced_works": [],
            },
            "https://openalex.org/W3": {
                "id": "https://openalex.org/W3",
                "title": "Empty Abstract",
                "abstract_inverted_index": None,
                "publication_year": 2023,
                "referenced_works": [],
            },
        }
        return payloads[openalex_id]


def test_fetch_referenced_documents_keeps_papers_without_abstract():
    service = OpenAlexPaperService(EmptyAbstractReferencedClient())
    raw_paper = {
        "id": "https://openalex.org/W1",
        "title": "Seed A",
        "abstract_inverted_index": {"Seed": [0], "A": [1]},
        "publication_year": 2024,
        "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
        "cited_by_count": 80,
    }

    cited_docs, raw_cited, paper_links, failed_ids = service.fetch_referenced_documents(raw_paper)

    assert failed_ids == []
    assert [doc.document_id for doc in cited_docs] == ["Has Abstract", "Empty Abstract"]
    assert [doc.abstract for doc in cited_docs] == ["Has Abstract", ""]
    assert [item["id"] for item in raw_cited] == ["https://openalex.org/W2", "https://openalex.org/W3"]
    assert {(link.source_paper_id, link.target_paper_id) for link in paper_links} == {
        ("Seed A", "Has Abstract"),
        ("Seed A", "Empty Abstract"),
    }


class PagedSeedClient:
    def __init__(self):
        self.calls = []

    def search_works(
        self,
        keyword,
        must_have_references,
        per_page,
        page=1,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        self.calls.append(
            {
                "keyword": keyword,
                "must_have_references": must_have_references,
                "per_page": per_page,
                "page": page,
                "min_cited_by_count": min_cited_by_count,
                "min_publication_year": min_publication_year,
            }
        )
        if page == 1:
            return [
                {
                    "id": f"https://openalex.org/W{i}",
                    "title": f"Seed {i}",
                    "abstract_inverted_index": {"Seed": [0]},
                    "publication_year": 2024,
                    "referenced_works": [],
                }
                for i in range(1, 21)
            ]
        if page == 2:
            return [
                {
                    "id": f"https://openalex.org/W{i}",
                    "title": f"Seed {i}",
                    "abstract_inverted_index": {"Seed": [0]},
                    "publication_year": 2024,
                    "referenced_works": [],
                }
                for i in range(21, 41)
            ]
        return []


def test_fetch_seed_documents_pages_through_openalex_when_limit_exceeds_twenty():
    client = PagedSeedClient()
    service = OpenAlexPaperService(client)

    seed_docs, raw_seeds = asyncio.run(
        service.fetch_seed_documents_async(
            keyword="idea graph",
            max_seed_papers=25,
            require_references=True,
        )
    )

    assert len(seed_docs) == 25
    assert len(raw_seeds) == 25
    assert [doc.document_id for doc in seed_docs] == [f"Seed {i}" for i in range(1, 26)]
    assert [call["per_page"] for call in client.calls] == [20, 20]
    assert [call["page"] for call in client.calls] == [1, 2]


def test_fetch_seed_documents_forwards_min_publication_year_to_seed_search():
    client = PagedSeedClient()
    service = OpenAlexPaperService(client)

    asyncio.run(
        service.fetch_seed_documents_async(
            keyword="idea graph",
            max_seed_papers=1,
            require_references=True,
            fetch_all=True,
            min_publication_year=2021,
        )
    )

    assert client.calls[0]["min_publication_year"] == 2021


class ExplicitSeedClient:
    def __init__(self):
        self.calls = []

    def get_work(self, openalex_id):
        self.calls.append(openalex_id)
        payloads = {
            "https://openalex.org/W1": {
                "id": "https://openalex.org/W1",
                "title": "First Seed",
                "abstract_inverted_index": {"First": [0], "seed": [1]},
                "publication_year": 2024,
                "referenced_works": [],
            },
            "https://openalex.org/W2": None,
            "https://openalex.org/W3": {
                "id": "https://openalex.org/W3",
                "title": "No Abstract Seed",
                "abstract_inverted_index": None,
                "publication_year": 2023,
                "referenced_works": [],
            },
            "https://openalex.org/W4": {
                "id": "https://openalex.org/W4",
                "title": "Fourth Seed",
                "abstract_inverted_index": {"Fourth": [0], "seed": [1]},
                "publication_year": 2022,
                "referenced_works": [],
            },
        }
        return payloads[openalex_id]


class RaisingExplicitSeedClient(ExplicitSeedClient):
    def get_work(self, openalex_id):
        if openalex_id == "https://openalex.org/W2":
            self.calls.append(openalex_id)
            raise RuntimeError("OpenAlex lookup failed")
        return super().get_work(openalex_id)


def test_fetch_openalex_seed_documents_skips_raised_lookup_and_records_failed_id(caplog):
    client = RaisingExplicitSeedClient()
    service = OpenAlexPaperService(client)

    seed_docs, raw_seeds, failed_ids = service.fetch_openalex_seed_documents(
        [
            "https://openalex.org/W1",
            "https://openalex.org/W2",
            "https://openalex.org/W4",
        ]
    )

    assert [doc.document_id for doc in seed_docs] == ["First Seed", "Fourth Seed"]
    assert [item["id"] for item in raw_seeds] == ["https://openalex.org/W1", "https://openalex.org/W4"]
    assert failed_ids == ["https://openalex.org/W2"]
    assert client.calls == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W4",
    ]
    assert "openalex:get_work_skipped id=https://openalex.org/W2" in caplog.text


def test_fetch_openalex_seed_documents_skips_missing_and_preserves_order():
    client = ExplicitSeedClient()
    service = OpenAlexPaperService(client)

    seed_docs, raw_seeds, failed_ids = service.fetch_openalex_seed_documents(
        [
            " https://openalex.org/W1 ",
            "https://openalex.org/W2",
            "https://openalex.org/W3",
            "https://openalex.org/W4",
        ]
    )

    assert [doc.document_id for doc in seed_docs] == ["First Seed", "No Abstract Seed", "Fourth Seed"]
    assert [doc.abstract for doc in seed_docs] == ["First seed", "", "Fourth seed"]
    assert [item["id"] for item in raw_seeds] == [
        "https://openalex.org/W1",
        "https://openalex.org/W3",
        "https://openalex.org/W4",
    ]
    assert failed_ids == ["https://openalex.org/W2"]
    assert client.calls == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W3",
        "https://openalex.org/W4",
    ]


def test_fetch_openalex_seed_documents_deduplicates_ids_and_skips_blanks():
    client = ExplicitSeedClient()
    service = OpenAlexPaperService(client)

    seed_docs, raw_seeds, failed_ids = asyncio.run(
        service.fetch_openalex_seed_documents_async(
            [
                "",
                " https://openalex.org/W1 ",
                "   ",
                "https://openalex.org/W1",
                "https://openalex.org/W4",
                " https://openalex.org/W4 ",
            ]
        )
    )

    assert [doc.document_id for doc in seed_docs] == ["First Seed", "Fourth Seed"]
    assert [item["id"] for item in raw_seeds] == ["https://openalex.org/W1", "https://openalex.org/W4"]
    assert failed_ids == []
    assert client.calls == ["https://openalex.org/W1", "https://openalex.org/W4"]
