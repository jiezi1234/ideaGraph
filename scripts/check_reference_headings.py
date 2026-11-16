from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.pdf_markdown import _REFERENCES_HEADING_RE


def find_reference_heading(markdown: str):
    match = _REFERENCES_HEADING_RE.search(markdown)
    if match is None:
        return None
    line_no = markdown.count("\n", 0, match.start()) + 1
    return line_no, match.group(0).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether Markdown files contain a heading matched by the current reference-strip rule."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data/testidea",
        help="Root directory to scan recursively for .md files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    markdown_paths = sorted(root.rglob("*.md"))
    matched: list[tuple[Path, int, str]] = []
    missing: list[Path] = []

    for markdown_path in markdown_paths:
        markdown = markdown_path.read_text(encoding="utf-8")
        heading = find_reference_heading(markdown)
        if heading is None:
            missing.append(markdown_path)
            continue
        line_no, text = heading
        matched.append((markdown_path, line_no, text))

    print(f"root: {root}")
    print(f"markdown_files: {len(markdown_paths)}")
    print(f"matched_reference_heading: {len(matched)}")
    print(f"missing_reference_heading: {len(missing)}")

    if missing:
        print("\nmissing:")
        for path in missing:
            print(path)

    if matched:
        print("\nmatched:")
        for path, line_no, text in matched:
            print(f"{path}:{line_no}: {text}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
