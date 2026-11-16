from src.interfaces.cli.build_graph import build_parser


def test_build_parser_accepts_keyword_and_seed_limit():
    parser = build_parser()
    args = parser.parse_args(["build-graph", "--keyword", "idea graph", "--max-seed-papers", "3"])

    assert args.command == "build-graph"
    assert args.keyword == "idea graph"
    assert args.max_seed_papers == 3


def test_build_parser_accepts_benchmark_accepted_ideas_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-accepted-ideas",
            "--output-dir",
            "exports/benchmark",
            "--max-papers",
            "7",
        ]
    )

    assert args.command == "benchmark-accepted-ideas"
    assert args.venue == "ICLR.cc/2026/Conference"
    assert args.query == "rag"
    assert args.max_papers == 7
    assert args.output_dir == "exports/benchmark"


def test_build_parser_accepts_ai_idea_bench_pilot_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare-ai-idea-bench-pilot",
            "--dataset-json",
            ".cache/ai_idea_bench/target_paper_data.json",
            "--count",
            "10",
            "--output-dir",
            "exports/ai-idea-bench-pilot",
        ]
    )

    assert args.command == "prepare-ai-idea-bench-pilot"
    assert args.dataset_json == ".cache/ai_idea_bench/target_paper_data.json"
    assert args.count == 10
    assert args.output_dir == "exports/ai-idea-bench-pilot"


def test_build_parser_accepts_chat_idea_seed_paper_id():
    parser = build_parser()
    args = parser.parse_args(
        [
            "chat-idea",
            "--query",
            "graph-based ideas",
            "--seed-paper-id",
            "Graph-guided discovery",
        ]
    )

    assert args.command == "chat-idea"
    assert args.query == "graph-based ideas"
    assert args.seed_paper_id == "Graph-guided discovery"
    assert args.generation_profile == "ideagraph"
    assert args.retrieval_mode == "graph"


def test_build_parser_accepts_chat_idea_generation_profile():
    parser = build_parser()
    args = parser.parse_args(
        [
            "chat-idea",
            "--query",
            "graph-based ideas",
            "--generation-profile",
            "ai_idea_bench",
            "--retrieval-mode",
            "chunk_vector",
        ]
    )

    assert args.command == "chat-idea"
    assert args.generation_profile == "ai_idea_bench"
    assert args.retrieval_mode == "chunk_vector"


def test_build_parser_accepts_repeated_openalex_seed_ids():
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-openalex-seed-graph",
            "--openalex-id",
            "W1",
            "--openalex-id",
            "https://openalex.org/W2",
        ]
    )

    assert args.command == "build-openalex-seed-graph"
    assert args.openalex_ids == ["W1", "https://openalex.org/W2"]


def test_build_parser_accepts_entity_extraction_preview_output_formats():
    parser = build_parser()
    json_args = parser.parse_args(
        [
            "preview-entity-extraction",
            "--md",
            "paper.md",
            "--output",
            "json",
        ]
    )

    table_args = parser.parse_args(
        [
            "preview-entity-extraction",
            "--md",
            "paper.md",
            "--output",
            "table",
        ]
    )

    assert json_args.command == "preview-entity-extraction"
    assert json_args.markdown_path == "paper.md"
    assert json_args.output == "json"
    assert table_args.command == "preview-entity-extraction"
    assert table_args.markdown_path == "paper.md"
    assert table_args.output == "table"
