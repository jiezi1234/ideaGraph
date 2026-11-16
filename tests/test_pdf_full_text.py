from pathlib import Path

from src.extraction.pdf_markdown import OpenDataLoaderPdfMarkdownExtractor, strip_references


def test_strip_references_removes_references_and_later_sections():
    markdown = """# Title

# Introduction

Useful body.

# References

[1] A cited paper.

# Appendix

Extra material.
"""

    assert strip_references(markdown).strip() == "# Title\n\n# Introduction\n\nUseful body."


def test_strip_references_matches_common_reference_heading_variants():
    cases = [
        "REFERENCES",
        " References",
        "# 6. REFERENCES",
        "## 8 REFERENCES",
        "# 6. Bibliography",
        "参考文献",
        "引用文献",
    ]

    for heading in cases:
        markdown = f"# Title\n\nUseful body.\n\n{heading}\n\n[1] A cited paper."

        assert strip_references(markdown).strip() == "# Title\n\nUseful body."


def test_opendataloader_extractor_uses_project_api_with_hybrid_then_fast_fallback(tmp_path, caplog):
    calls = []

    def converter(**kwargs):
        calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "paper.md"
        json_path = output_dir / "paper.json"
        if kwargs.get("hybrid") == "docling-fast":
            output_path.write_text("# Title\n\npartial", encoding="utf-8")
            json_path.write_text('{"kids": [{"page number": 1}]}', encoding="utf-8")
            return None
        output_path.write_text("# Title\n\nfull text\n\n# References\n\nrefs", encoding="utf-8")
        json_path.write_text('{"kids": [{"page number": 1}, {"page number": 2}]}', encoding="utf-8")
        return None

    extractor = OpenDataLoaderPdfMarkdownExtractor(
        converter=converter,
        hybrid_url="http://127.0.0.1:15002",
    )

    result = extractor.extract(
        pdf_path=tmp_path / "paper.pdf",
        output_dir=tmp_path / "out",
        expected_page_count=2,
    )

    assert result.mode == "fast"
    assert result.markdown_path.read_text(encoding="utf-8").strip() == "# Title\n\nfull text"
    assert [call.get("hybrid") for call in calls] == ["docling-fast", None]
    assert calls[0]["hybrid_url"] == "http://127.0.0.1:15002"
    assert calls[0]["format"] == "markdown,json"
    assert calls[0]["image_output"] == "off"
    assert "pdf_extract:hybrid_incomplete" in caplog.text


def test_opendataloader_extractor_falls_back_to_fast_when_hybrid_errors(tmp_path, caplog):
    calls = []

    def converter(**kwargs):
        calls.append(kwargs)
        if kwargs.get("hybrid") == "docling-fast":
            raise RuntimeError("hybrid backend unavailable")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper.md").write_text("# Title\n\nbody\n\n# Bibliography\n\nrefs", encoding="utf-8")
        (output_dir / "paper.json").write_text('{"number of pages": 1, "kids": [{"page number": 1}]}', encoding="utf-8")

    extractor = OpenDataLoaderPdfMarkdownExtractor(
        converter=converter,
        hybrid_url="http://127.0.0.1:15002",
    )

    result = extractor.extract(
        pdf_path=tmp_path / "paper.pdf",
        output_dir=tmp_path / "out",
        expected_page_count=1,
    )

    assert result.mode == "fast"
    assert result.markdown_path.read_text(encoding="utf-8").strip() == "# Title\n\nbody"
    assert [call.get("hybrid") for call in calls] == ["docling-fast", None]
    assert "pdf_extract:hybrid_failed" in caplog.text
