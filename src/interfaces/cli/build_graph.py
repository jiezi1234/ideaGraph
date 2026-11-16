import argparse

from src.agent.generation_profiles import GENERATION_PROFILE_CHOICES

RETRIEVAL_MODE_CHOICES = ("graph", "paper_vector", "chunk_vector")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-graph")
    build.add_argument("--keyword", required=True)
    build.add_argument("--max-seed-papers", type=int, default=3)
    build.add_argument("--require-references", action="store_true", default=False)
    
    chat = subparsers.add_parser("chat-idea")
    chat.add_argument("--query", required=True)
    chat.add_argument("--seed-paper-id", default=None)
    chat.add_argument("--archive-output-dir", default=None)
    chat.add_argument("--generation-profile", choices=GENERATION_PROFILE_CHOICES, default="ideagraph")
    chat.add_argument("--retrieval-mode", choices=RETRIEVAL_MODE_CHOICES, default="graph")

    benchmark = subparsers.add_parser("benchmark-accepted-ideas")
    benchmark.add_argument("--venue", default="ICLR.cc/2026/Conference")
    benchmark.add_argument("--query", default="rag")
    benchmark.add_argument("--max-papers", type=int, default=50)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument(
        "--decision-scope",
        choices=["accepted", "rejected", "mixed", "all"],
        default="accepted",
    )

    ai_idea_bench = subparsers.add_parser("prepare-ai-idea-bench-pilot")
    ai_idea_bench.add_argument("--dataset-json", required=True)
    ai_idea_bench.add_argument("--count", type=int, default=10)
    ai_idea_bench.add_argument("--output-dir", required=True)

    openalex_seed = subparsers.add_parser("build-openalex-seed-graph")
    openalex_seed.add_argument("--openalex-id", action="append", dest="openalex_ids", required=True)

    local_paper = subparsers.add_parser("build-local-paper-graph")
    local_paper.add_argument("--paper-dir", action="extend", nargs="+", dest="paper_dirs", required=True)

    preview = subparsers.add_parser("preview-entity-extraction")
    preview.add_argument("--md", dest="markdown_path", required=True)
    preview.add_argument("--output", choices=["json", "table"], default="json")
    return parser
