中文版本见 [README.zh-CN.md](README.zh-CN.md)。

## MVP Goal

Build a Neo4j paper citation graph from OpenAlex, optionally extract semantic
entities and relations, then use the local graph to generate grounded research
ideas.

## Quick Start

1. Install `uv` from https://docs.astral.sh/uv/getting-started/installation/.
2. From the project root, install dependencies with `uv sync`.
3. Copy `.env.example` to `.env` and fill in your local values.
4. Run commands with `uv run`, for example:

```bash
uv run python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

## Configuration

Copy `.env.example` to `.env` and fill in your local values.

## Run

```bash
uv run python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

When full-text extraction is enabled, start the OpenDataLoader hybrid backend in
a separate shell first:

```bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
CUDA_VISIBLE_DEVICES=1 uv run opendataloader-pdf-hybrid --host 127.0.0.1 --port 15002 --device cuda --log-level info
```

Then run graph building with the normal application command:

```bash
FULL_TEXT_EXTRACTION_ENABLED=true \
PDF_HYBRID_URL=http://127.0.0.1:15002 \
uv run python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

```bash
uv run python main.py chat-idea --query "graph-based scientific idea generation for low-resource domains"
```

### Occasional Workflows

Run the local bus-scheduling idea-generation comparison across all three
systems. The first command prepares shared local-paper inputs for
AI-Researcher and ResearchAgent; it also writes the same commands to
`exports/testbus-local-baselines/manifest.json` for reruns. See
`docs/ai-researcher-researchagent-bus-scheduling-runbook.md` for the complete
local comparison runbook:

```bash
cd /home/zb/code/IdeaGraph

QUESTION="利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"
TARGET_PAPER_DIR="/home/zb/code/IdeaGraph/data/testbus/A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling"
OUTPUT_DIR="/home/zb/code/IdeaGraph/exports/testbus-local-baselines"

python3 scripts/prepare_local_baseline_inputs.py \
  --paper-dir "$TARGET_PAPER_DIR" \
  --question "$QUESTION" \
  --output-dir "$OUTPUT_DIR" \
  --category bus_sched \
  --instance-id bus_evo_nn
```

Run IdeaGraph on the same local paper. `build-local-paper-graph` uses the
paper-folder name as the local `Paper.id`; pass that same id to `chat-idea`
when you want this exact paper to ground generation:

```bash
cd /home/zb/code/IdeaGraph

uv run python main.py build-local-paper-graph \
  --paper-dir "$TARGET_PAPER_DIR"

uv run python main.py chat-idea \
  --query "$QUESTION" \
  --seed-paper-id "A Novel Evolutionary Neural Network-based Approach for Online Bus Scheduling" \
  --archive-output-dir "$OUTPUT_DIR/ideagraph"
```

Run AI-Researcher through the local-paper idea-agent adapter. Replace
`API_BASE_URL` if Ollama is exposed on a different host or port:

```bash
cd /home/zb/code/IdeaGraph

API_BASE_URL=http://localhost:11435 \
CHEEP_MODEL=ollama_chat/qwen3.6:35b \
/home/zb/code/AI-Researcher/.venv/bin/python \
  scripts/run_ai_researcher_local_idea_agent.py \
  --ai-researcher-root /home/zb/code/AI-Researcher \
  --input-dir "$OUTPUT_DIR/ai_researcher" \
  --category bus_sched \
  --instance-id bus_evo_nn \
  --task-level task2
```

Run ResearchAgent through the local-context adapter. This skips OpenAlex
reference expansion and feeds the prepared local paper context directly into
ResearchAgent's three-stage idea pipeline:

```bash
cd /home/zb/code/IdeaGraph

PYTHONPATH=/home/zb/code/ResearchAgent/code \
/home/zb/code/ResearchAgent/.venv/bin/python \
  scripts/run_researchagent_local_context.py \
  --context-jsonl "$OUTPUT_DIR/researchagent/local_contexts.jsonl" \
  --results-path "$OUTPUT_DIR/researchagent/ideas.jsonl" \
  --researchagent-code-dir /home/zb/code/ResearchAgent/code \
  --model-name qwen3.6:35b \
  --iterations 3
```

Prepare and evaluate a small AI Idea Bench target-paper-aligned pilot. The
pilot keeps target papers out of the graph, builds a reference-paper graph,
runs `chat-idea` on reviewed research problems, scores generated ideas with the
official-style I2I motivation / experiment-plan judge, and measures retrieved
reference-paper hit rate:

```bash
python3 scripts/run_ai_idea_bench_queries.py \
  --problem-drafts exports/ai-idea-bench-lidar-perception-10/problem_drafts_v2.jsonl \
  --archive-output-dir exports/ai-idea-bench-lidar-perception-10/chat_idea_runs_v2 \
  --runs-output exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --query-field research_problem \
  --limit 10
```

```bash
python3 scripts/evaluate_ai_idea_bench_i2i.py \
  --judge-mode official_i2i \
  --dataset-json .cache/ai_idea_bench/Idea_bench_data/target_paper_data.json \
  --run-manifest exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --output-json exports/ai-idea-bench-lidar-perception-10/i2i_official_v2.json \
  --output-csv exports/ai-idea-bench-lidar-perception-10/i2i_official_v2.csv
```

```bash
python3 scripts/evaluate_ai_idea_bench_retrieval_hits.py \
  --run-manifest exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --reference-availability exports/ai-idea-bench-lidar-perception-10/reference_availability.json \
  --output-json exports/ai-idea-bench-lidar-perception-10/retrieval_hits_v2.json \
  --output-csv exports/ai-idea-bench-lidar-perception-10/retrieval_hits_v2.csv
```

See `docs/ai-idea-bench-evaluation.md` for the full pilot workflow and output
contracts.

Normalize offline idea-generation outputs from IdeaGraph, ResearchAgent, and
AI-Researcher before blind pairwise judging. The normalized `method_idea_v1`
boundary keeps the judge decoupled from system-specific output formats. When a
normalized idea contains `original`, the judge sees only that `Original` text;
otherwise it sees only `Title`, `Summary`, `Problem`, `Method`, and
`Experiment`. AI-Researcher outputs are normalized by reading the selected
`Idea X` from `selected_idea` and using `ideas[X-1]` as `original`; missing or
out-of-range indices fail fast:

```bash
python3 scripts/normalize_method_idea_outputs.py \
  --input exports/chat_idea_runs/20260604-055159-query-c194029a3dcd.json \
  --output-json exports/testbus-method-comparison/normalized/ideagraph.json \
  --method-name IdeaGraph \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"

python3 scripts/normalize_method_idea_outputs.py \
  --input exports/testbus-local-baselines/researchagent/ideas.jsonl \
  --output-json exports/testbus-method-comparison/normalized/researchagent.json \
  --method-name ResearchAgent \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"

python3 scripts/normalize_method_idea_outputs.py \
  --input exports/testbus-local-baselines/ai_researcher/outputs/local_idea_agent/ideas_bus_sched_bus_evo_nn.json \
  --output-json exports/testbus-method-comparison/normalized/ai_researcher.json \
  --method-name AI-Researcher \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？"
```

Run blind comparison over all normalized methods. With three inputs, the script
creates all pairwise combinations and evaluates each pair twice with reversed
`Idea A` / `Idea B` order:

```bash
python3 scripts/compare_normalized_method_ideas.py \
  --idea-json exports/testbus-method-comparison/normalized/ideagraph.json \
  --idea-json exports/testbus-method-comparison/normalized/researchagent.json \
  --idea-json exports/testbus-method-comparison/normalized/ai_researcher.json \
  --question "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？" \
  --output-json exports/testbus-method-comparison/blind_comparison_3methods.json \
  --output-md exports/testbus-method-comparison/blind_comparison_3methods.md
```

Build a benchmark from OpenReview venue papers, convert each paper into the same
`IdeaOutput` schema used by `chat-idea`, evaluate it with the existing judge
prompt, and export JSON/CSV/HTML results:

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --decision-scope accepted \
  --output-dir exports/iclr2026-rag-benchmark
```

The output directory contains `results.json`, `results.csv`, and an interactive
`report.html` that shows per-paper converted ideas, score breakdowns, radar
charts, ICLR-score correlations by evaluation dimension, and a status filter
for accepted/rejected/other papers.
If `results.json` already exists, the workflow treats it as an incremental
baseline: papers with the same OpenReview ID are overwritten by the new run,
and papers not touched by the new run remain in all regenerated outputs.

To inspect rejected papers, switch the decision scope:

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --decision-scope rejected \
  --output-dir exports/iclr2026-rag-benchmark-rejected
```

Open the HTML report directly to inspect benchmark results:

```bash
xdg-open exports/iclr2026-rag-benchmark/report.html
```

For command-line inspection, use `results.csv` for compact scores and
`results.json` for full converted ideas plus evaluator details:

```bash
head -n 5 exports/iclr2026-rag-benchmark/results.csv
```

Build a graph from one or more explicit OpenAlex works. This command expands
exactly one reference hop from the provided seed works, then uses the existing
full-text and entity extraction settings for downstream graph construction.
Explicit seed works are kept as `Paper` nodes even when OpenAlex cannot provide
an abstract or resolvable full text; those contentless seed nodes still expand
citations but skip entity extraction until they are later enriched:

```bash
uv run python main.py build-openalex-seed-graph \
  --openalex-id https://openalex.org/Wxxxx \
  --openalex-id Wyyyy
```

Build a graph from one or more local paper folders. Each folder must contain a
PDF; the folder name becomes the paper `document_id`, and the abstract is
extracted from the paper PDF by the model before the existing entity extraction
and graph writing flow runs:

```bash
uv run python main.py build-local-paper-graph \
  --paper-dir /path/to/paper-a /path/to/paper-b
```

Preview entity extraction for a Markdown paper without writing to Neo4j. Use
`--output json` for the normalized machine-readable shape, or `--output table`
for terminal tables of the same entity and relation fields:

```bash
uv run python main.py preview-entity-extraction \
  --md /path/to/paper.md \
  --output json
```

The preview command always prints two top-level results: `full_text` runs the
whole Markdown baseline, while `incremental` extracts the abstract first and
then maintains entities/relations across reference-stripped tokenizer chunks.

After graph construction finishes, query the built graph with `chat-idea`:

```bash
uv run python main.py chat-idea --query "linear graph retrieval augmented generation"
```

If you already know the graph paper that should ground the generation, pass its
exact `Paper.id` and `chat-idea` will skip query understanding, embedding
retrieval, and paper selection:

```bash
uv run python main.py chat-idea \
  --query "linear graph retrieval augmented generation" \
  --seed-paper-id "Graph-guided discovery"
```

Each `chat-idea` run writes a JSON archive for later inspection under
`exports/chat_idea_runs/` by default. Override the destination per run when
needed:

```bash
uv run python main.py chat-idea \
  --query "linear graph retrieval augmented generation" \
  --archive-output-dir exports/my-query-runs
```

To export the current Neo4j graph as GraphML:

```bash
uv run python export_graphml.py
```

To upload the current Neo4j graph to Graphistry for interactive visualization,
keep `GRAPHISTRY_PERSONAL_KEY_ID` and `GRAPHISTRY_PERSONAL_KEY_SECRET` in
`.env`, then run:

```bash
uv run python scripts/visualize_graphistry.py --name "IdeaGraph full graph"
```

The script reads the same `NEO4J_*` settings as the main app and binds
Graphistry's Bolt session to `NEO4J_DATABASE`, so it does not accidentally query
Neo4j's default database. Omit `--limit` to upload the full graph, including
isolated nodes; use `--limit 1000` for a smaller first smoke test. Set
`GRAPHISTRY_SERVER` or pass `--graphistry-server` only when using a non-default
Graphistry server.

## Current Behavior

- configuration is loaded from `.env`
- citation expansion supports `one_hop` and `priority_expand`
- OpenAlex requests can use an optional comma-separated API key pool
- network failures are skipped instead of aborting the whole run
- Neo4j database is created automatically if it does not exist
- when enabled, paper PDFs are resolved from OpenAlex external locations first,
  OpenAlex `content_urls.pdf` second, and arXiv title lookup last. Existing
  Markdown artifacts under `data/{NEO4J_DATABASE}/{paper}/` are reused before
  any download; successful attempts are stored as `source_N.pdf` plus
  `attempt_N/*.md` and used for abstract recovery plus entity/relation
  extraction
- full-text entity extraction is abstract-first and incremental: it initializes
  graph state from the abstract, strips References/Bibliography from the
  Markdown body, then processes tokenizer-aligned chunks as patch updates
- local paper folder imports use the folder name as `document_id`, rebuild any
  existing paper with that id, preserve existing incoming/outgoing `CITES`
  edges, extract the abstract from the PDF with the model, and then reuse the
  same entity and graph-writing flow
- if full text is unavailable, entities and relations are extracted from title
  and abstract
- `preview-entity-extraction` is a no-write comparison workflow: it emits both
  whole-Markdown extraction and abstract-first incremental chunk extraction,
  each with only `entities` and `relations` fields
- entities and relations include short descriptions; extraction logs model input
  and output token counts when the model provider returns usage metadata
- when `ENTITY_RESOLUTION_ENABLED=true`, extracted entities are aligned only
  against existing Neo4j entities. Existing `Entity.id` values stay stable;
  aliases are accumulated, and resolver-produced merged descriptions replace
  stale entity descriptions instead of appending duplicate text
- after relation endpoints are canonicalized, existing entity-to-entity
  relations with the same source, relation type, and target can have their
  descriptions merged by the model. Relation `source_refs` accumulate across
  papers, and model failures fall back to append-and-dedupe writes
- each paper is extracted and written to Neo4j incrementally
- citation edges among fetched cited papers are also created when present
- paper ids are normalized to readable paper titles when OpenAlex titles exist
- `chat-idea` rewrites user queries into a `retrieval_query` plus
  `query_keywords`; the paper retrieval node uses `retrieval_query` for
  ChromaDB-backed embedding candidate retrieval, while `query_keywords` are
  passed to the agent as graph-tool planning context rather than being searched
  automatically. Chroma indexes are persisted under
  `.cache/chroma/{NEO4J_DATABASE}` and are rebuilt from Neo4j papers/entities
  when stale. Chunk-vector retrieval uses paragraph-first Markdown chunks with
  `CHUNK_MIN_TOKENS=256`, `CHUNK_MAX_TOKENS=2048`, and a local HuggingFace
  tokenizer configured by `CHUNK_TOKENIZER_MODEL_PATH` so chunk lengths match
  the generation-context model more closely than whitespace counting. The agent
  loop can call `get_node_neighborhood` for one-hop graph
  context before returning a structured Chinese idea organized as
  `Problem -> Method -> Experiment`
- when `--seed-paper-id` is provided, `chat-idea` loads that exact `Paper.id`
  from Neo4j as the selected paper and continues from agent generation, without
  running query understanding, embedding retrieval, or paper selection
- `chat-idea` uses Ollama's OpenAI-compatible chat completions API for model
  calls and graph-tool `message.tool_calls`.
- when the graph-tool loop stops returning tool calls, `chat-idea` first tries
  to parse that assistant response directly as `IdeaOutput` JSON and only falls
  back to a separate `final_idea_generation` structured call when the content
  is empty or invalid
- `chat-idea` graph-tool lookup first tries exact `Paper.title` or `Entity.id`,
  then falls back to case-insensitive `Entity.aliases`; graph-tool node payloads
  are compacted to the properties needed by generation
- `chat-idea` then runs a single-model idea evaluation node over only the
  generated idea and prints an overall score, per-criterion scores, diagnosis,
  and revision instruction
- `chat-idea` archives each query result as JSON, including query understanding,
  retrieved candidates, selected paper, graph-tool trace, merged graph context,
  generated idea, evaluation, and optional thinking trace
- `chat-idea` can also write Langfuse Cloud traces when `LANGFUSE_PUBLIC_KEY`
  and `LANGFUSE_SECRET_KEY` are set; model calls are traced through
  `langfuse.openai.OpenAI`, while retrieval, selection, agent, and graph-tool
  steps remain workflow observations. For round-level debugging, the workflow
  also records `agent_generation_round_N_result` observations containing the
  returned `content`, `reasoning`, `finish_reason`, and normalized
  `{input, output, total}` usage
- when the Ollama model exposes `message.thinking`, `chat-idea` prints the final
  idea-generation thinking trace for debugging
- prompt templates for query understanding, paper-to-idea conversion, idea
  agent behavior, entity extraction, and idea evaluation live as Python modules
  under `src/prompts/`; import them through `src.prompts` instead of reading
  `.txt` files
- structured model outputs are enforced in code through OpenAI-compatible
  `structured_json` calls with Pydantic schemas; prompts should describe the
  task and constraints rather than duplicate full JSON schema blocks
- GraphML export and `chat-idea` query archives write generated files under
  `exports/`; generated exports are local artifacts and should not be committed
  unless explicitly requested
- `scripts/visualize_graphistry.py` uploads the current Neo4j graph to
  Graphistry through Bolt. It reads Graphistry credentials from `.env`, keeps the
  selected Neo4j database scoped to `NEO4J_DATABASE`, and prints the Graphistry
  URL through the normal project logger.

To backfill normalized lowercase entity aliases in an existing Neo4j database:

```bash
uv run python scripts/backfill_entity_aliases.py --dry-run
uv run python scripts/backfill_entity_aliases.py
```

See `docs/full-text-extraction.md` for full-text extraction configuration and
runtime notes.
