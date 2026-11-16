#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCH_DIR="${BENCH_DIR:-exports/ai-idea-bench-lidar-perception-10}"
REFERENCE_PAPER_DIRS="${REFERENCE_PAPER_DIRS:-$BENCH_DIR/reference_paper_dirs_by_title.txt}"
PROBLEM_DRAFTS="${PROBLEM_DRAFTS:-$BENCH_DIR/problem_drafts.jsonl}"
CHAT_ARCHIVE_DIR="${CHAT_ARCHIVE_DIR:-$BENCH_DIR/chat_idea_runs}"
RUNS_OUTPUT="${RUNS_OUTPUT:-$BENCH_DIR/runs.jsonl}"
FAILURES_OUTPUT="${FAILURES_OUTPUT:-$BENCH_DIR/failures.jsonl}"
LOG_DIR="${LOG_DIR:-$BENCH_DIR/pipeline_logs}"

QUERY_LIMIT="${QUERY_LIMIT:-10}"
QUERY_FIELD="${QUERY_FIELD:-focused_problem}"
QUERY_VARIANT="${QUERY_VARIANT:-ideagraph_ai_bench}"
QUERY_FORCE="${QUERY_FORCE:-0}"

TESTBUS_DATABASE="${TESTBUS_DATABASE:-testbus}"
TESTBUS_KEYWORD="${TESTBUS_KEYWORD:-Evolutionary Neural Networks bus scheduling}"
TESTBUS_MAX_SEED_PAPERS="${TESTBUS_MAX_SEED_PAPERS:-100}"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR" "$CHAT_ARCHIVE_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S)-pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "required file not found: $path" >&2
    exit 1
  fi
}

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_file "$REFERENCE_PAPER_DIRS"
require_file "$PROBLEM_DRAFTS"

reference_count="$(wc -l < "$REFERENCE_PAPER_DIRS" | tr -d ' ')"
problem_count="$(wc -l < "$PROBLEM_DRAFTS" | tr -d ' ')"

echo "log_file=$LOG_FILE"
echo "reference_paper_dirs=$REFERENCE_PAPER_DIRS count=$reference_count"
echo "problem_drafts=$PROBLEM_DRAFTS count=$problem_count"
echo "step1_database=current-env"
echo "step2_database=current-env"
echo "step3_database=$TESTBUS_DATABASE strategy=priority_expand"

# Step 1: build the AI Idea Bench LiDAR reference graph in the current env database.
run_cmd xargs -r -d $'\n' -a "$REFERENCE_PAPER_DIRS" \
  python3 main.py build-local-paper-graph --paper-dir

# Step 2: run 10 AI Idea Bench-style queries against the current env database.
query_cmd=(
  python3 scripts/run_ai_idea_bench_queries.py
  --problem-drafts "$PROBLEM_DRAFTS"
  --archive-output-dir "$CHAT_ARCHIVE_DIR"
  --runs-output "$RUNS_OUTPUT"
  --failures-output "$FAILURES_OUTPUT"
  --variant "$QUERY_VARIANT"
  --query-field "$QUERY_FIELD"
  --limit "$QUERY_LIMIT"
)
if [[ "$QUERY_FORCE" == "1" ]]; then
  query_cmd+=(--force)
fi
run_cmd "${query_cmd[@]}"

# Step 3: rebuild the testbus keyword graph with priority expansion.
run_cmd env \
  NEO4J_DATABASE="$TESTBUS_DATABASE" \
  CITATION_GRAPH_STRATEGY=priority_expand \
  python3 main.py build-graph \
  --keyword "$TESTBUS_KEYWORD" \
  --max-seed-papers "$TESTBUS_MAX_SEED_PAPERS"

echo
echo "pipeline finished"
