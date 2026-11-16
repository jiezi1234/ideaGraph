# AI Idea Bench Evaluation

This repo includes a lightweight AI Idea Bench adapter for target-paper-aligned
idea-generation experiments. It is intended for small controlled pilots where
target papers are excluded from the graph, reference papers are used as the
retrieval corpus, `chat-idea` generates ideas, and a separate judge compares the
generated idea against the target paper.

## Inputs

- AI Idea Bench target data JSON, for example
  `.cache/ai_idea_bench/Idea_bench_data/target_paper_data.json`.
- A target-selection export directory under `exports/`, typically containing:
  - `selected_targets.json`
  - `reference_availability.json`
  - `reference_corpus_manifest.json`
  - `problem_drafts.jsonl` or manually reviewed `problem_drafts_v2.jsonl`
  - `reference_paper_dirs_by_title.txt`
- A Neo4j database that contains only the reference-paper graph for the target
  set. Do not build target papers into the graph being evaluated.

`exports/` artifacts are local run outputs and should not be committed unless a
review explicitly asks for them.

## Select Targets

Use `scripts/select_ai_idea_bench_targets.py` to select related target papers
and export target metadata plus reference-corpus manifests. The script supports
TF-IDF selection and an Ollama embedding backend with cache.

```bash
python3 scripts/select_ai_idea_bench_targets.py \
  --dataset-json .cache/ai_idea_bench/Idea_bench_data/target_paper_data.json \
  --output-dir exports/ai-idea-bench-lidar-perception-10 \
  --count 10
```

For formal scoring, manually review or regenerate the benchmark-facing problem
questions. The recommended field for reviewed questions is top-level
`research_problem`; the older `sanitized_problem.focused_problem` field is kept
for smoke tests.

## Build Reference Graph

Local reference-paper imports should use title-based paper directories so folder
indices do not leak into `Paper.id`:

```bash
xargs -r -d $'\n' -a exports/ai-idea-bench-lidar-perception-10/reference_paper_dirs_by_title.txt \
  python3 main.py build-local-paper-graph --paper-dir
```

## Run Queries

Run `chat-idea` against each problem draft and write a run manifest. Use
`research_problem` for manually reviewed questions:

```bash
python3 scripts/run_ai_idea_bench_queries.py \
  --problem-drafts exports/ai-idea-bench-lidar-perception-10/problem_drafts_v2.jsonl \
  --archive-output-dir exports/ai-idea-bench-lidar-perception-10/chat_idea_runs_v2 \
  --runs-output exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --failures-output exports/ai-idea-bench-lidar-perception-10/failures_v2.jsonl \
  --query-field research_problem \
  --limit 10
```

The run manifest is a JSONL file with `dataset_index`, `target_paper`, `query`,
`archive_path`, `variant`, and `generation_profile`.

## Official I2I Scoring

`scripts/evaluate_ai_idea_bench_i2i.py` defaults to `--judge-mode official_i2i`.
This mode mirrors the official AI Idea Bench I2I shape:

- compare generated motivation against target motivation in one judge call
- compare generated experiment plan against target experiment plan in another
  judge call
- return `rating` and `explanation` for each dimension
- do not pass paper titles to the judge
- do not report a primary overall score

```bash
python3 scripts/evaluate_ai_idea_bench_i2i.py \
  --judge-mode official_i2i \
  --dataset-json .cache/ai_idea_bench/Idea_bench_data/target_paper_data.json \
  --run-manifest exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --output-json exports/ai-idea-bench-lidar-perception-10/i2i_official_v2.json \
  --output-csv exports/ai-idea-bench-lidar-perception-10/i2i_official_v2.csv
```

For internal A/B checks only, `--judge-mode local` keeps the older combined
judge with an `overall_score`. Do not compare local-mode scores directly with
official AI Idea Bench tables.

The judge model comes from the current `.env` settings. To compare strictly
against published AI Idea Bench numbers, align the judge model with the official
evaluation setup as well as the prompt shape.

## Retrieval Hit Rate

Use `scripts/evaluate_ai_idea_bench_retrieval_hits.py` to measure whether each
query retrieved the target paper's gold reference papers. The script collects
paper titles from `candidate_papers`, `selected_paper`, `tool_trace`, and final
`graph_context`, then title-normalizes them against
`reference_availability.json`.

```bash
python3 scripts/evaluate_ai_idea_bench_retrieval_hits.py \
  --run-manifest exports/ai-idea-bench-lidar-perception-10/runs_v2.jsonl \
  --reference-availability exports/ai-idea-bench-lidar-perception-10/reference_availability.json \
  --output-json exports/ai-idea-bench-lidar-perception-10/retrieval_hits_v2.json \
  --output-csv exports/ai-idea-bench-lidar-perception-10/retrieval_hits_v2.csv
```

The main metrics are:

- `hit_rate`: matched gold references divided by gold reference count.
- `retrieval_precision`: matched gold references divided by retrieved paper
  count after title de-duplication.

## One-Off Pipeline

`scripts/run_ai_idea_bench_lidar_pipeline.sh` is a local convenience pipeline
for the LiDAR pilot. It builds the local reference graph, runs the configured
problem drafts, and then rebuilds the separate `testbus` graph. Override paths
and query fields through environment variables such as `PROBLEM_DRAFTS`,
`QUERY_FIELD`, `CHAT_ARCHIVE_DIR`, and `RUNS_OUTPUT`.
