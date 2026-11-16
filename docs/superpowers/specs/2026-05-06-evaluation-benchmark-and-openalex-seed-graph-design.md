# ICLR Accepted Paper Benchmark and OpenAlex Seed Graph Design

## Goal

Add two occasional-use workflows without changing the current `build-graph` and
`chat-idea` main flows:

1. Build a benchmark from ICLR 2026 RAG-related papers, convert each paper into
   the same `IdeaOutput` schema used by normal idea generation, score those
   converted ideas with the existing evaluator, and export files plus a chart
   for inspection.
2. Build a graph from one or more user-specified OpenAlex works, using those
   works as explicit seeds instead of keyword search, then run the existing
   paper/citation/full-text/entity workflow over the seed papers and their
   one-hop references.

## Non-Goals

- Do not change `chat-idea` retrieval, generation, tool-call, or evaluation
  behavior.
- Do not change `build-graph --keyword` behavior.
- Do not introduce the future "build graph from existing local documents"
  workflow in this change.
- Do not make OpenReview or arXiv access mandatory for the existing graph build
  path.

## Workflow 1: OpenReview Paper Benchmark

Add a separate benchmark workflow, for example
`src/workflows/accepted_paper_benchmark.py`, and keep it independent from the
agent chat graph.

The benchmark command should be:

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --decision-scope accepted \
  --output-dir exports/iclr2026-rag-benchmark
```

The workflow should:

1. Fetch public OpenReview submissions for the venue.
2. Support accepted, rejected, mixed, and all decision scopes.
3. Filter for RAG-related papers using a configurable text query over title,
   abstract, keywords, and available metadata.
4. Try to resolve a PDF for each paper:
   - Prefer OpenReview PDF/content links when available.
   - Fall back to arXiv title lookup through the existing `ArxivClient`.
5. Convert PDF to Markdown through the existing PDF full-text path.
6. Apply the existing `strip_references()` filter before any model step.
7. Convert the paper into `IdeaOutput`.
8. Score the `IdeaOutput` through the existing `idea_evaluation` function and
   `src/prompts/evaluate_research_idea.py`.
9. Export machine-readable results and a simple static report.

The paper-to-idea conversion should use the dedicated benchmark prompt module
`src/prompts/convert_paper_to_idea.py`. The model call must use
OpenAI-compatible `structured_json` with the `IdeaOutput` schema instead of
duplicating the full JSON schema in the prompt. Its output must validate
against these `IdeaOutput` fields:

- `title`
- `summary`
- `problem`
- `method`
- `core_paper_id`
- `core_paper_title`

The conversion input should prefer `full_text_markdown` after reference removal.
If no PDF/full text is available, it should fall back to title and abstract and
record that fallback in the output metadata.

The exported files should be:

- `results.json`: complete benchmark records, including OpenReview metadata,
  PDF/full-text status, converted `IdeaOutput`, evaluator output, and any
  available ICLR review score. If the file already exists, it is the incremental
  baseline: new records overwrite matching `openreview_id` entries, and untouched
  existing records remain in the regenerated outputs.
- `results.csv`: compact table for sorting and plotting.
- `report.html`: static chart/report for manual inspection, including a
  client-side status filter.

ICLR reviewer scores should be best-effort:

- Parse publicly available review ratings if OpenReview exposes them.
- If ratings are unavailable or inconsistent, leave them null.
- Support `--review-score-csv` to allow manually supplied scores to override or
  fill missing values. The minimum useful columns are `openreview_id`, `title`,
  and `iclr_score`.

## Workflow 2: OpenAlex Seed Graph

Add a separate OpenAlex-specific seed workflow and CLI command. The command name
should make the OpenAlex source explicit and avoid reserving names needed by a
future local-document graph builder:

```bash
uv run python main.py build-openalex-seed-graph \
  --openalex-id https://openalex.org/Wxxxx \
  --openalex-id Wyyyy
```

This workflow should use one or more OpenAlex works as explicit seeds. It should
not perform keyword search. It should expand exactly one reference hop from
those seeds and should not use `CITATION_MAX_TOTAL_PAPERS` or the
`priority_expand` loop. The total graph scope is:

- all successfully fetched seed works
- all successfully fetched works directly referenced by those seeds
- `CITES` edges among those known papers when both endpoints exist

After the seed and one-hop cited papers are known, the workflow should reuse the
same downstream behavior as the current graph build:

- write `Paper` nodes
- write `CITES` edges after endpoint nodes exist
- optionally enrich papers with PDF-derived Markdown when
  `FULL_TEXT_EXTRACTION_ENABLED=true`
- strip references from full-text Markdown through the existing PDF/full-text
  path
- optionally extract entities and relations when
  `EXTRACT_ENTITIES_ENABLED=true`
- write `CONTAINS` and entity `REL` edges

The intended implementation shape is to factor shared producer/consumer logic in
`MvpGraphBuildWorkflow` enough that both keyword seed discovery and explicit
OpenAlex seed discovery can reuse the same writing and extraction path without
duplicating the worker logic.

## Data and Error Handling

- OpenReview, arXiv, OpenAlex, and PDF failures should be logged and skipped per
  paper where possible.
- A missing benchmark PDF should not drop the paper if title and abstract are
  available.
- An OpenAlex seed that cannot be fetched should be logged and skipped; other
  seed IDs should still proceed.
- The benchmark should record status fields so missing ICLR scores or missing
  PDFs are visible in the report instead of hidden.
- Existing standard-library logging should be used; no `print` inside workflow
  internals.

## Testing

Add focused tests for:

- CLI parsing for `benchmark-accepted-ideas`.
- CLI parsing for repeated `--openalex-id` values.
- Decision filtering keeps accepted, rejected, mixed, and all scopes visible in
  the report and testable in the workflow.
- Paper-to-idea conversion validates against `IdeaOutput`.
- Benchmark scoring calls the existing evaluator with only `idea_output`.
- Benchmark full-text processing applies `strip_references()`.
- Explicit OpenAlex seed graph writes seed papers, one-hop cited papers, and
  citations.
- Explicit OpenAlex seed graph does not invoke keyword search or
  `priority_expand`.
- Full-text enrichment still occurs before entity extraction in the OpenAlex
  seed graph path.

Before completion, run:

```bash
python3 -m pytest -q
```
