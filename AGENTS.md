# IdeaGraph Agent Notes

## Purpose and Boundaries
- This repo is the MVP for building a paper citation graph plus optional
  entity/relation extraction.
- Primary graph-build flow: OpenAlex seed search -> `one_hop` or
  `priority_expand` citation expansion -> `Paper` nodes and `CITES` edges ->
  optional PDF Markdown full text -> entity/relation extraction ->
  `Paper-[:CONTAINS]->Entity` and `Entity-[:REL]->Entity`.
- `producer` owns paper discovery plus paper/citation graph construction.
- `consumer` owns entity extraction plus semantic graph construction.
- Do not move `CITES` writing into the consumer path without re-checking node
  existence ordering.
- Pending citations should be tracked as pending edges, not by re-queueing full
  papers.
- `chat-idea` is the read/query workflow. It must not mutate graph build state.

## Coding Standards
- Use standard library `logging`, not `print`. Logging is configured in
  `src/logging_config.py` and writes to both console and file.
- 写代码时要加必要的中文注释：重点解释非显而易见的业务意图、图谱写入顺序、
  prompt/schema 契约、重试/容错逻辑、以及容易被后续维护误改的边界条件。
- 中文注释要短而准；不要给简单赋值、直观分支、普通参数传递写无意义注释。
- Keep normal package imports at module top level. Do not hide project
  dependencies inside helper methods such as `_load_*`; missing imports should
  fail visibly.

## CLI Workflows
- Keep occasional CLI workflows separate from the main graph-build and
  `chat-idea` flows.
- `benchmark-accepted-ideas` fetches OpenReview venue papers through API v2
  `content.venueid`, filters by `--decision-scope`
  (`accepted|rejected|mixed|all`), converts papers to `IdeaOutput`, scores them
  with the existing evaluator, and incrementally writes `results.json`,
  `results.csv`, and `report.html`. Existing records with the same
  `openreview_id` are overwritten; untouched records remain in exported outputs.
- `build-openalex-seed-graph` takes explicit OpenAlex work IDs, skips keyword
  search, expands exactly one reference hop, then reuses the full-text/entity
  graph construction path. Explicit seed papers are kept as `Paper` nodes and
  used for citation expansion even when they still lack abstract and full text;
  in that case they skip entity extraction until later enrichment.
- `build-local-paper-graph` takes local paper folders, uses the folder name as
  `document_id`, extracts an abstract from the PDF when needed, rebuilds any
  existing paper with the same id while preserving `CITES` edges, then reuses
  the existing entity extraction and graph-writing flow.
- `preview-entity-extraction` takes a Markdown paper through the entity
  extraction prompt/schema path, normalizes the would-be graph entities and
  relations, prints JSON or terminal tables with only entity/relation fields,
  and must not connect to or write Neo4j. It always emits both `full_text`
  and `incremental` results; the incremental result extracts the abstract
  first, strips references before tokenizer-aligned body chunks, and carries
  current entities/relations forward as compact `[field]----[field]` state.
- AI Idea Bench scripts are benchmark adapters, not main workflows. Keep target
  papers out of the graph, use reviewed `research_problem` prompts for formal
  runs, score I2I with `--judge-mode official_i2i`, and treat retrieval hit
  rate as a separate reference-paper recall diagnostic.
- Method-comparison scripts are offline benchmark adapters. Keep raw-system
  readers in `scripts/normalize_method_idea_outputs.py` /
  `scripts/method_idea_format.py`; `scripts/compare_normalized_method_ideas.py`
  must consume only normalized `method_idea_v1` JSON. Blind judging uses
  `original` when present, otherwise only Title/Summary/Problem/Method/Experiment.

## Chat-Idea Contract
- `chat-idea --seed-paper-id <Paper.id>` loads that exact Neo4j `Paper.id` and
  skips query understanding, embedding retrieval, and paper selection.
- Normal flow is `agent_generation` before `idea_evaluation`. When evaluation is
  enabled, run one improvement pass: evaluate initial idea -> regenerate with
  feedback plus readable previous `graph_context` / `tool_trace` -> evaluate the
  improved idea. Do not reuse previous raw OpenAI message history.
- The evaluator receives only generated `idea_output`, never the user query,
  selected paper, graph context, or hidden retrieval state.
- `agent_generation` uses the configured OpenAI-compatible `/v1/chat/completions`
  endpoint (DashScope/Qwen by default) with real `message.tool_calls`. Keep assistant tool calls and graph
  tool results in one OpenAI message history; do not reintroduce JSON fallback
  protocols such as `{"tool_calls":[]}`.
- Graph tool loops are capped at 10 action rounds. Round 0 must call
  `get_node_neighborhood` to start graph exploration. The current fixed-path
  policy also requires rounds 1 and 2 to continue from returned neighbor nodes
  with `GRAPH_REACT_WIDTH` parallel directions when enough candidates exist.
  After the required graph exploration path, no `message.tool_calls` means final
  idea response: first parse assistant `message.content` as `IdeaOutput` JSON,
  and only fall back to `final_idea_generation` when content is empty or
  invalid.
- Agent generation round names and logs are zero-based:
  `agent_generation_round_0`, `agent_generation_round_1`, and so on.
- Graph tool execution failures must be returned as structured, recoverable
  tool results so the next round can fix invalid arguments.
- `chat-idea` must include structured `tool_trace` in workflow results and
  archived JSON. Archives belong under `CHAT_IDEA_ARCHIVE_DIR` or
  `--archive-output-dir` inside `exports/`; do not commit them.
- Langfuse tracing uses `langfuse.openai.OpenAI` for model observations and
  token usage when credentials are set. Graph tool execution remains a separate
  tool observation. Round debug observations use
  `agent_generation_round_N_result`.
- Final idea generation may include the model's `message.thinking` as
  `idea_thinking` for debugging when the selected model exposes it.
- `get_node_neighborhood.node_types` must use only `Paper` or concrete semantic
  entity labels such as `Concept`, `Domain`, `Problem`, `Method`, `Scenario`,
  or `Innovation`; never use base `Entity` as a filter value. Exact/alias misses
  may fall back to the vector-most-related non-Paper entity and must include
  structured `lookup_notes`.
- Graph tools and vector retrieval must not return or index `Paper` nodes with
  missing or blank `abstract`; citation-only papers may remain in Neo4j for
  `CITES` structure, but they are not useful evidence for `chat-idea`.
- `src/prompts/chat_idea_agent.py` must preserve the graph-grounded
  `Problem -> Method -> Experiment` structure: selected paper as target paper,
  candidate/neighbor papers as related papers, semantic graph nodes as entities.

## Structured Model and Prompts
- Prompt templates are Python modules under `src/prompts/`; each exposes
  `PROMPT`, and `src.prompts` re-exports named constants. Do not reintroduce
  `.txt` prompt files.
- Query understanding, paper-to-idea conversion, idea generation, idea
  evaluation, abstract extraction, and entity extraction must use
  OpenAI-compatible `structured_json` with Pydantic schemas. Do not reintroduce
  `OllamaJsonModel` / `generate_messages_json` JSON-mode production paths.
- Evaluation prompt module: `src/prompts/evaluate_research_idea.py`. Keep it
  focused on role, task, scoring, and language requirements with XML-like tags
  such as `<role>`, `<context>`, `<task>`, `<scoring_rules>`,
  `<evaluation_criteria>`, and `<output_requirements>`.
- Do not add a tool-strategy section unless the evaluator is actually allowed
  to use tools. Evaluation output schema uses `IdeaEvaluation` and
  `EvaluationCriterion` in `src/schemas/agent.py`.
- AI Idea Bench official I2I uses separate motivation and experiment-plan
  judges with `rating`/`explanation` fields. Do not pass paper titles to that
  judge or collapse the official score into a primary `overall_score`.

## Extraction Contract
- Entity extraction uses PDF-derived Markdown full text when
  `FULL_TEXT_EXTRACTION_ENABLED=true` and a PDF can be resolved; otherwise use
  title + abstract. OpenAlex abstracts are used directly when present; missing
  abstracts may be extracted from PDF-derived full text before graph writing.
- Production full-text entity extraction uses the same abstract-first,
  reference-stripped, tokenizer-aligned incremental chunk flow as the
  `preview-entity-extraction` incremental result. Do not reintroduce a separate
  whole-full-text production extraction path.
- Prompt modules: `src/prompts/extract_paper_entities.py` and
  `src/prompts/extract_paper_abstract.py`.
- Entity and abstract extractors must call `structured_json` with Pydantic
  output schemas. Keep field/format schemas in code, not duplicated in prompt
  text.
- Entity and relation resolution also use `structured_json` through the shared
  structured model wrapper, so `LLM_CACHE_ENABLED=true` caches resolver LLM
  calls as well as extraction calls.
- Entity outputs must use exact fields `entity_id`, `entity_type`, and
  `description`. Relation outputs must use exact schema fields, including
  `description`. Do not accept legacy aliases such as `name`, `text`, `source`,
  `target`, `head_entity_id`, or `tail_entity_id`.
- `entity_type` must be one of the exact allowed values; do not coerce or guess
  from partial matches. Do not reintroduce a separate `name` field without a
  clear schema change.
- `entity_id` and relation endpoints must stay natural-language phrases with
  normal spaces, not snake_case, camelCase, or slug identifiers unless that is
  the official term spelling.
- Entity IDs should stay cross-paper mergeable. Do not extract paper-internal
  layer ordinals, stage numbers, ablation labels, or local implementation
  placeholders as standalone entities unless they are reusable scientific
  concepts or named methods; put local roles into relation descriptions.
- If model output schema is invalid, log a warning and retry with the validation
  reason. Extraction results may include model input/output token counts when
  available.

## Full-Text PDF Handling
- Full-text Markdown is produced through the `opendataloader-pdf` Python API.
  Hybrid mode is attempted first; incomplete or failed hybrid extraction falls
  back to fast mode and logs a warning.
- Full-text enrichment first reuses any existing non-empty Markdown artifact
  under `data/{NEO4J_DATABASE}/{paper}/`; a cache hit must skip PDF download.
- OpenAlex PDF candidates should be tried as external locations first,
  `content_urls.pdf` last with the configured API-key pool, and arXiv title
  lookup only after OpenAlex candidates fail. Each candidate must pass download,
  `%PDF` validation, and Markdown conversion before it is accepted.
- The hybrid backend is not started by application code. Run it separately when
  full-text extraction is enabled:

  ```bash
  uv run opendataloader-pdf-hybrid --host 127.0.0.1 --port 15002 --device cuda --log-level info
  ```

- Full-text outputs under `data/{NEO4J_DATABASE}/{paper}/` and local-paper
  outputs under `data/{NEO4J_DATABASE}/local_papers/{paper}/` are local
  artifacts and are not committed.
- Local paper folder imports also use `opendataloader-pdf` before abstract
  extraction and downstream graph construction.

## Graph Model
- Paper nodes use label `Paper`.
- `Paper.id` and `Entity.id` must have uniqueness constraints in Neo4j.
- Entity nodes use base label `Entity` plus the type label from `entity_type`,
  for example `:Entity:Concept`.
- Entity writes must include a normalized lowercase `Entity.id` in `aliases` so
  graph-tool lookup can fall back from exact id matching to case-insensitive
  alias matching.
- Paper-to-paper edges use `:CITES`; paper-to-entity edges use `:CONTAINS`.
- Entity-to-entity relation types must remain Neo4j-safe. Prompts may ask for
  underscore relation names such as `Used_For`; the repository also normalizes
  relation names with `.upper().replace("-", "_")` before Cypher.
- Entity resolution only aligns current-paper entities to existing Neo4j
  entities. Do not use resolver cache entries created during the same
  extraction as candidates before they are written successfully.
- Existing `Entity.id` values are stable. When resolution matches an existing
  node, keep that id, merge aliases, and replace the entity description only
  with the resolver's grounded `merged_description`.
- Relation resolution runs only after relation endpoints are canonicalized and
  only for the exact `(source_entity_id, relation_type, target_entity_id)` key.
  It may merge relation descriptions, but must not change endpoints or relation
  type; `source_refs` must accumulate across papers.
- Graph tool responses should expose only compact properties needed by the
  agent: `Paper` nodes return `id` and `abstract`; `Entity` nodes return `id`,
  `entity_type`, and `description`.

## External Source Behavior
- Keyword paper search should use an OpenAlex filter such as
  `title_and_abstract.search:{keyword}` and should not require
  `has_abstract:true`; missing abstracts are filled later from PDF-derived full
  text when available. Keyword seed papers that still have neither abstract nor
  full text after preparation are skipped before graph writing.
- Citation-expanded papers that still have neither abstract nor full text are
  kept as `Paper`/`CITES` graph structure but skipped for entity extraction;
  `priority_expand` may still continue expanding from their OpenAlex raw
  reference data.
- Citation graph strategy is configurable:
  - `one_hop`: keep current one-hop expansion.
  - `priority_expand`: seed search with a cited-by threshold, then keep
    expanding by cited count priority until the total paper limit.
- OpenAlex concurrent cited-paper fetches are limited locally in
  `OpenAlexPaperService`.
- OpenAlex keyword searches should log request URL, page, per-page, and filter
  before the HTTP request is sent. Do not log API key values.
- Network/API failures should be logged and skipped, not crash the entire run.
- OpenReview note list responses are consumed from the standard `notes` array;
  do not depend on a `results` fallback shape.

## Runtime Configuration
- Main config is loaded from `.env` through `src/config/settings.py`.
- Graph build: `CITATION_GRAPH_STRATEGY`,
  `CITATION_SEED_MIN_CITED_BY_COUNT`,
  `CITATION_SEED_MIN_PUBLICATION_YEAR`, `CITATION_MAX_TOTAL_PAPERS`,
  `OPENALEX_WORKS_FILTER`, `OPENALEX_API_KEYS`,
  `OPENALEX_API_KEY_COOLDOWN_SECONDS`.
- Extraction/PDF: `EXTRACT_ENTITIES_ENABLED`, `EXTRACTION_WORKER_COUNT`,
  `ENTITY_RESOLUTION_ENABLED`, `FULL_TEXT_EXTRACTION_ENABLED`,
  `FULL_TEXT_OUTPUT_DIR`, `PDF_HYBRID_URL`, `PDF_HYBRID_BACKEND`,
  `PDF_HYBRID_MODE`.
- Agent/model/cache: `AGENT_RETRIEVAL_TOP_K`, `CHAT_IDEA_ARCHIVE_DIR`,
  `GRAPH_MULTIHOP_DEPTH`, `GRAPH_REACT_WIDTH`, `LLM_*`, `EMBED_*`. Chunk-vector
  retrieval uses `CHUNK_TOKENIZER_MODEL_PATH`,
  `CHUNK_TOKENIZER_LOCAL_FILES_ONLY`, `CHUNK_MIN_TOKENS`,
  `CHUNK_MAX_TOKENS`, and `CHUNK_OVERLAP_TOKENS` for tokenizer-aligned
  paragraph chunking. `chat-idea` uses a ChromaDB vector index under
  `.cache/chroma/{NEO4J_DATABASE}` for paper/entity/chunk retrieval.
- Storage/tracing/logging: `NEO4J_*`, `LANGFUSE_*`, `LOG_LEVEL`, `LOG_FILE`.
- Graphistry visualization: `GRAPHISTRY_PERSONAL_KEY_ID`,
  `GRAPHISTRY_PERSONAL_KEY_SECRET`, optional `GRAPHISTRY_SERVER`.

## Local Files
- Do not commit `.env`, `.cache/`, `logs/`, `.codex/`, `.run/`, `exports/`,
  `data/`, `paper_db/`, `__pycache__/`, or `tool_planning_prompt.txt`.

## Validation
- Run `python3 -m pytest -q` before claiming code changes are complete.
- If changing prompt/schema/workflow behavior, update or add tests in `tests/`.
