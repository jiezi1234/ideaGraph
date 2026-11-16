from src.ingestion.openreview_client import (
    OpenReviewClient,
    OpenReviewPaper,
    extract_review_scores,
    filter_accepted_papers,
    filter_papers_by_decision_scope,
    filter_papers_by_query,
    normalize_openreview_note,
    _response_notes,
)


def test_filter_accepted_papers_keeps_only_final_accept_statuses():
    papers = [
        OpenReviewPaper(openreview_id="a", title="A", abstract="RAG", status="ICLR 2026 Oral"),
        OpenReviewPaper(openreview_id="b", title="B", abstract="RAG", status="ICLR 2026 Poster"),
        OpenReviewPaper(openreview_id="c", title="C", abstract="RAG", status="ICLR 2026 Spotlight"),
        OpenReviewPaper(
            openreview_id="d",
            title="D",
            abstract="RAG",
            status="ICLR 2026 Conference Desk Rejected Submission",
        ),
        OpenReviewPaper(openreview_id="e", title="E", abstract="RAG", status="Submitted to ICLR 2026"),
        OpenReviewPaper(
            openreview_id="f",
            title="F",
            abstract="RAG",
            status="ICLR 2026 Conference Withdrawn Submission",
        ),
        OpenReviewPaper(openreview_id="g", title="G", abstract="RAG", status="ICLR 2026 Rejected"),
    ]

    accepted = filter_accepted_papers(papers)

    assert [paper.openreview_id for paper in accepted] == ["a", "b", "c"]


def test_filter_papers_by_decision_scope_supports_rejected_and_mixed_runs():
    papers = [
        OpenReviewPaper(openreview_id="a", title="A", abstract="RAG", status="Accept (Oral)"),
        OpenReviewPaper(openreview_id="b", title="B", abstract="RAG", status="Reject"),
        OpenReviewPaper(openreview_id="c", title="C", abstract="RAG", status="Accept (Poster)"),
        OpenReviewPaper(openreview_id="d", title="D", abstract="RAG", status="Submitted to ICLR 2026"),
    ]

    assert [paper.openreview_id for paper in filter_papers_by_decision_scope(papers, "accepted")] == ["a", "c"]
    assert [paper.openreview_id for paper in filter_papers_by_decision_scope(papers, "rejected")] == ["b"]
    assert [paper.openreview_id for paper in filter_papers_by_decision_scope(papers, "mixed")] == ["a", "b", "c"]
    assert [paper.openreview_id for paper in filter_papers_by_decision_scope(papers, "all")] == ["a", "b", "c", "d"]


def test_normalize_openreview_note_prefers_decision_over_venue_for_status():
    note = {
        "id": "note-1",
        "content": {
            "title": {"value": "Paper title"},
            "abstract": {"value": "Paper abstract"},
            "decision": {"value": "Reject"},
            "venue": {"value": "ICLR 2026 Conference"},
            "venueid": {"value": "ICLR.cc/2026/Conference"},
        },
    }

    paper = normalize_openreview_note(note, base_url="https://openreview.net")

    assert paper.status == "Reject"


def test_filter_papers_by_query_matches_title_abstract_keywords_and_status():
    papers = [
        OpenReviewPaper(
            openreview_id="a",
            title="Retrieval augmented generation",
            abstract="",
            keywords=[],
            status="ICLR 2026 Poster",
        ),
        OpenReviewPaper(
            openreview_id="b",
            title="Vision model",
            abstract="No retrieval here",
            keywords=["rag"],
            status="ICLR 2026 Poster",
        ),
        OpenReviewPaper(
            openreview_id="c",
            title="Optimizer",
            abstract="Training curves",
            keywords=[],
            status="ICLR 2026 Poster",
        ),
        OpenReviewPaper(
            openreview_id="d",
            title="Calibration",
            abstract="",
            keywords=[],
            status="ICLR 2026 RAG Poster",
        ),
        OpenReviewPaper(
            openreview_id="e",
            title="Diffusion",
            abstract="",
            keywords=[],
            status="ICLR 2026 Poster",
            metadata={"tag": "retrieval-only-metadata"},
        ),
        OpenReviewPaper(
            openreview_id="f",
            title="Bragging Models",
            abstract="",
            keywords=[],
            status="ICLR 2026 Poster",
        ),
    ]

    matched = filter_papers_by_query(papers, "rag retrieval")

    assert [paper.openreview_id for paper in matched] == ["a", "b", "d"]
    assert filter_papers_by_query(papers, " - ") == papers


def test_filter_papers_by_query_can_include_metadata_when_enabled():
    papers = [
        OpenReviewPaper(
            openreview_id="a",
            title="Paper title",
            abstract="Paper abstract",
            keywords=[],
            status="ICLR 2026 Poster",
            metadata={"tags": ["rag"]},
        ),
        OpenReviewPaper(
            openreview_id="b",
            title="Paper title",
            abstract="Paper abstract",
            keywords=[],
            status="ICLR 2026 Poster",
            metadata={"tags": ["other"]},
        ),
    ]

    matched = filter_papers_by_query(papers, "rag", include_metadata=True)

    assert [paper.openreview_id for paper in matched] == ["a"]


def test_normalize_openreview_note_reads_content_fields_and_pdf_url():
    note = {
        "id": "note-1",
        "content": {
            "title": {"value": "Paper title"},
            "abstract": {"value": "Paper abstract"},
            "keywords": {"value": ["RAG", "LLM"]},
            "venue": {"value": "ICLR 2026 Poster"},
            "pdf": {"value": "/pdf?id=note-1"},
        },
    }

    paper = normalize_openreview_note(note, base_url="https://openreview.net")

    assert paper.openreview_id == "note-1"
    assert paper.title == "Paper title"
    assert paper.abstract == "Paper abstract"
    assert paper.keywords == ["RAG", "LLM"]
    assert paper.status == "ICLR 2026 Poster"
    assert paper.pdf_url == "https://openreview.net/pdf?id=note-1"
    assert paper.metadata == note


def test_extract_review_scores_reads_numeric_rating_values():
    notes = [
        {"content": {"rating": {"value": "8: accept, good paper"}, "score": {"value": "2: weak"}}},
        {"content": {"recommendation": {"value": "6: marginally above acceptance threshold"}}},
        {"content": {"score": {"value": 7}}},
        {"content": {"rating": {"value": "not numeric"}}},
    ]

    scores = extract_review_scores(notes)

    assert scores == [8.0, 6.0, 7.0]


def test_response_notes_ignores_results_payload():
    class FakeResponse:
        def json(self):
            return {"results": [{"id": "note-1"}]}

    assert _response_notes(FakeResponse()) == []


def test_fetch_venue_papers_uses_openreview_notes_endpoint():
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeHttpClient:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None):
            self.calls.append((url, params))
            if url.endswith("/groups"):
                return FakeResponse(
                    {
                        "groups": [
                            {
                                "content": {
                                    "rejected_venue_id": {"value": "ICLR.cc/2026/Conference/Rejected_Submission"},
                                    "withdrawn_venue_id": {"value": "ICLR.cc/2026/Conference/Withdrawn_Submission"},
                                    "desk_rejected_venue_id": {"value": "ICLR.cc/2026/Conference/Desk_Rejected_Submission"},
                                }
                            }
                        ]
                    }
                )
            if params and params.get("content.venueid") == "ICLR.cc/2026/Conference":
                return FakeResponse(
                    {
                        "notes": [
                            {
                                "id": "note-1",
                                "content": {
                                    "title": {"value": "Accepted paper"},
                                    "abstract": {"value": "RAG"},
                                    "venue": {"value": "ICLR 2026 Oral"},
                                    "venueid": {"value": "ICLR.cc/2026/Conference"},
                                },
                            }
                        ]
                    }
                )
            if params and params.get("content.venueid") == "ICLR.cc/2026/Conference/Rejected_Submission":
                return FakeResponse(
                    {
                        "notes": [
                            {
                                "id": "note-2",
                                "content": {
                                    "title": {"value": "Rejected paper"},
                                    "abstract": {"value": "RAG"},
                                    "venue": {"value": "Submitted to ICLR 2026"},
                                    "venueid": {"value": "ICLR.cc/2026/Conference/Rejected_Submission"},
                                },
                            }
                        ]
                    }
                )
            return FakeResponse({"notes": []})

    http_client = FakeHttpClient()
    client = OpenReviewClient(http_client=http_client)

    papers = client.fetch_venue_papers("ICLR.cc/2026/Conference")

    assert [paper.openreview_id for paper in papers] == ["note-1", "note-2"]
    assert http_client.calls[0][0].endswith("/groups")
    assert http_client.calls[1][0].endswith("/notes")
    assert http_client.calls[1][1]["content.venueid"] == "ICLR.cc/2026/Conference"
    assert http_client.calls[2][0].endswith("/notes")
    assert http_client.calls[2][1]["content.venueid"] == "ICLR.cc/2026/Conference/Rejected_Submission"
