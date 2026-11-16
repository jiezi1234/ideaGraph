# Preview Incremental Entity Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a preview-only comparison between whole-text entity extraction and abstract-first incremental extraction.

**Architecture:** `EntityExtractionPreviewWorkflow` orchestrates both strategies and keeps output formatting local to the preview feature. `PaperExtractor` gains optional current-state inputs for incremental patch extraction without changing normal extraction callers. Preview chunking reuses the same tokenizer-backed paragraph chunking as chunk-vector retrieval.

**Tech Stack:** Python, Pydantic schemas, existing `PaperExtractor`, existing `merge_entities` / `normalize_relations`, pytest.

---

### Task 1: Lock Preview Output Behavior

**Files:**
- Modify: `tests/test_entity_extraction_preview.py`

- [x] **Step 1: Write the failing test**

Add a test that creates Markdown with title, Abstract, Method, and References
sections. Assert that `workflow.run()` returns top-level `full_text` and
`incremental` keys, that the full-text call sees the complete Markdown, that
the incremental abstract call sees only the abstract, and that the body chunk
does not include text after `## References`.

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest -q tests/test_entity_extraction_preview.py tests/test_paper_extractor.py tests/test_prompt_loading.py
```

Expected before implementation: failures for missing incremental calls and old
single-result table formatting.

### Task 2: Add Extractor Current-State Input

**Files:**
- Modify: `src/extraction/paper_extractor.py`
- Modify: `tests/test_paper_extractor.py`

- [x] **Step 1: Write the failing test**

Add a test that calls `PaperExtractor.extract(document, existing_entities=[...],
existing_relations=[...])` and asserts that the user message includes
`<current_entities>`, `<current_relations>`, and an instruction to return only
changed entities and relations.

- [x] **Step 2: Implement minimal support**

Thread optional `existing_entities` and `existing_relations` into the prompt
payload. Serialize only schema fields needed by the model in compact
`[field]----[field]` line format:
`entity_id`, `entity_type`, `description`, `source_entity_id`,
`target_entity_id`, `relation_type`.

### Task 3: Implement Dual Preview Workflow

**Files:**
- Modify: `src/workflows/entity_extraction_preview.py`
- Modify: `src/prompts/extract_paper_entities.py`

- [x] **Step 1: Preserve old full-text baseline**

Keep the existing full-Markdown extraction as the `full_text` result.

- [x] **Step 2: Add incremental extraction**

Extract the Abstract section, run abstract extraction first when present, strip
references from the Markdown, remove the Abstract section from body processing,
split body text with the shared tokenizer-backed paragraph chunk helper, and
process chunks sequentially.

- [x] **Step 3: Apply local patch state each round**

After the abstract extraction, run `merge_entities` and `normalize_relations`
to initialize state. For each chunk, interpret the returned `entities` and
`relations` as a patch: add missing entries, overwrite `description` for
existing entries, preserve existing entity ids/types, and skip relations whose
endpoints are unavailable. Pass the updated normalized entities and relations
into the next chunk.

- [x] **Step 4: Update table formatting**

Render separate `Full Text` and `Incremental` table sections while preserving
the existing entity/relation columns.

### Task 4: Verify

**Files:**
- Test: `tests/test_entity_extraction_preview.py`
- Test: `tests/test_paper_extractor.py`
- Test: `tests/test_prompt_loading.py`

- [x] **Step 1: Run targeted tests**

Run:

```bash
python3 -m pytest -q tests/test_entity_extraction_preview.py tests/test_paper_extractor.py tests/test_prompt_loading.py
```

Expected after implementation: all targeted tests pass.

- [x] **Step 2: Run full suite**

Run:

```bash
python3 -m pytest -q
```

Expected: repository test suite passes.
