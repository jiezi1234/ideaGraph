# Vector-Only RAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `paper_vector` and `chunk_vector` retrieval modes so `chat-idea` can run graph-enabled and pure-vector baselines independently from the generation output profile.

**Architecture:** Keep `generation_profile` as the prompt/output-schema axis and add `retrieval_mode` as the retrieval-context axis. Extend the existing Chroma backend with a deterministic paragraph-first chunk index and route `chunk_vector` generation through structured output with retrieved chunks injected into the prompt, while disabling graph tools.

**Tech Stack:** Python 3.12, Pydantic, ChromaDB, LangGraph, pytest, existing Ollama OpenAI-compatible model wrapper.

---

### Task 1: Add Chunk Data Structures and Chroma Chunk Retrieval

**Files:**
- Modify: `src/schemas/document.py`
- Modify: `src/retrieval/base.py`
- Modify: `src/retrieval/chroma_retriever.py`
- Test: `tests/test_chroma_retriever.py`

- [ ] **Step 1: Write failing chunk index tests**

Add tests that create `RetrievedPaperDocument` records with `full_text_markdown_path`, verify `ChromaVectorIndex` creates `chunks`, verifies paragraph metadata, and verifies `ChromaChunkRetriever.retrieve()` returns sorted chunk records with `paper_id`, `chunk_id`, `chunk_index`, `paragraph_start`, `paragraph_end`, `score`, `text_source`, and `text`.

Run: `python3 -m pytest tests/test_chroma_retriever.py -q`
Expected: FAIL because chunk fields, `ChromaChunkRetriever`, and chunk sync do not exist.

- [ ] **Step 2: Implement chunk schemas and retriever protocol**

Add `RetrievedContextChunk` to `src/schemas/document.py`. Extend `RetrievedPaperDocument` with optional `full_text_markdown_path` and `text_source`. Add `ContextChunkRetriever` protocol in `src/retrieval/base.py`.

- [ ] **Step 3: Implement paragraph-first chunking and Chroma `chunks` sync**

Add a `chunks` collection to `ChromaVectorIndex`. Implement deterministic chunk creation with `min_tokens=120`, `target_tokens=(450, 700)`, `max_tokens=2000`, `overlap_tokens=80`, paper-bound metadata, hash-based skip, and stale deletion.

- [ ] **Step 4: Verify chunk retrieval tests pass**

Run: `python3 -m pytest tests/test_chroma_retriever.py -q`
Expected: PASS.

### Task 2: Add Repository Projection for Vector Indexing

**Files:**
- Modify: `src/storage/graph/neo4j_repository.py`
- Test: `tests/test_neo4j_repository.py`

- [ ] **Step 1: Write failing repository projection test**

Add a test for `list_papers_for_vector_index()` that returns papers with abstract, cited count, `full_text_markdown_path`, and `text_source`.

Run: `python3 -m pytest tests/test_neo4j_repository.py::test_repository_lists_papers_for_vector_index_with_full_text_path -q`
Expected: FAIL because the method does not exist.

- [ ] **Step 2: Implement repository method**

Add `list_papers_for_vector_index()` and keep `list_papers_with_abstracts()` behavior unchanged. The new method should include papers with non-empty abstract or non-empty `full_text_markdown_path`.

- [ ] **Step 3: Verify repository tests pass**

Run: `python3 -m pytest tests/test_neo4j_repository.py::test_repository_lists_papers_for_vector_index_with_full_text_path -q`
Expected: PASS.

### Task 3: Route `retrieval_mode` Through `chat-idea`

**Files:**
- Modify: `src/agent/state.py`
- Modify: `src/agent/graph.py`
- Modify: `src/workflows/idea_chat.py`
- Modify: `src/app/bootstrap.py`
- Test: `tests/test_idea_chat_workflow.py`

- [ ] **Step 1: Write failing workflow tests**

Add tests that `paper_vector` disables graph tools and produces no `tool_trace`, and that `chunk_vector` calls the chunk retriever, injects retrieved chunks into the structured generation prompt, and archives `retrieval_mode` plus `retrieved_chunks`.

Run: `python3 -m pytest tests/test_idea_chat_workflow.py -q`
Expected: FAIL because `retrieval_mode` and chunk retrieval routing do not exist.

- [ ] **Step 2: Implement runtime routing**

Add `retrieval_mode` to `IdeaChatWorkflow` and `build_idea_chat_graph`. For `graph`, preserve current behavior. For `paper_vector`, pass no graph provider. For `chunk_vector`, retrieve chunks after paper selection and include a `Retrieved vector context` section in `_generate_idea_without_tools()`.

- [ ] **Step 3: Verify workflow tests pass**

Run: `python3 -m pytest tests/test_idea_chat_workflow.py -q`
Expected: PASS.

### Task 4: Add CLI, Archive, Batch Runner, and Retrieval-Hit Support

**Files:**
- Modify: `src/interfaces/cli/build_graph.py`
- Modify: `main.py`
- Modify: `src/workflows/query_archive.py`
- Modify: `scripts/run_ai_idea_bench_queries.py`
- Modify: `scripts/evaluate_ai_idea_bench_retrieval_hits.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_main.py`
- Test: `tests/test_run_ai_idea_bench_queries_script.py`
- Test: `tests/test_ai_idea_bench_retrieval_hits_script.py`

- [ ] **Step 1: Write failing CLI and archive tests**

Add tests for `--retrieval-mode`, workflow builder routing, run manifest recording, and retrieval-hit extraction from `retrieved_chunks`.

Run: `python3 -m pytest tests/test_cli.py tests/test_main.py tests/test_run_ai_idea_bench_queries_script.py tests/test_ai_idea_bench_retrieval_hits_script.py -q`
Expected: FAIL because `retrieval_mode` is not parsed, forwarded, or archived.

- [ ] **Step 2: Implement CLI and script plumbing**

Parse `--retrieval-mode graph|paper_vector|chunk_vector`, pass it to `build_idea_chat_workflow`, archive `retrieval_mode`, forward it through `scripts/run_ai_idea_bench_queries.py`, and let retrieval-hit extraction include paper titles from `retrieved_chunks`.

- [ ] **Step 3: Verify CLI and script tests pass**

Run: `python3 -m pytest tests/test_cli.py tests/test_main.py tests/test_run_ai_idea_bench_queries_script.py tests/test_ai_idea_bench_retrieval_hits_script.py -q`
Expected: PASS.

### Task 5: Full Verification and Cleanup

**Files:**
- Modify only if tests expose integration gaps.

- [ ] **Step 1: Run focused tests**

Run: `python3 -m pytest tests/test_chroma_retriever.py tests/test_neo4j_repository.py tests/test_idea_chat_workflow.py tests/test_cli.py tests/test_main.py tests/test_run_ai_idea_bench_queries_script.py tests/test_ai_idea_bench_retrieval_hits_script.py -q`
Expected: PASS.

- [ ] **Step 2: Run full suite**

Run: `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 3: Run diff checks**

Run: `git diff --check`
Expected: no output.
