from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Sequence


LOGGER = logging.getLogger(__name__)


def load_contexts(context_jsonl: Path) -> list[dict]:
    contexts = []
    for line_number, line in enumerate(context_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        context = json.loads(line)
        if not isinstance(context.get("paper"), dict):
            raise ValueError(f"line {line_number}: missing paper object")
        context.setdefault("references", [])
        context.setdefault("entities", [])
        contexts.append(context)
    return contexts


def run_local_contexts(
    *,
    context_jsonl: Path,
    results_path: Path,
    pipeline_factory: Callable[[], object],
) -> list[dict]:
    contexts = load_contexts(context_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    pipeline = pipeline_factory()
    for context in contexts:
        title = context.get("paper", {}).get("title", "")
        LOGGER.info("researchagent_local:start title=%s", title)
        result = pipeline.run(context)
        results.append(result)
        with results_path.open("a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        LOGGER.info("researchagent_local:done title=%s", title)
    return results


def build_researchagent_pipeline(
    *,
    researchagent_code_dir: Path,
    model_name: str | None,
    iterations: int,
):
    code_dir = researchagent_code_dir.expanduser().resolve()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    env_path = code_dir.parent / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        LOGGER.warning("researchagent_local:dotenv_unavailable env_path=%s", env_path)

    # ResearchAgent 本地上下文 runner 只复用生成管线，不调用它的 OpenAlex 检索入口。
    from models.ollama import OllamaClient
    from pipelines.research_pipeline import ResearchPipeline

    client = OllamaClient(model=model_name)
    return ResearchPipeline(api_client=client, iterations=iterations)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResearchAgent on local paper contexts without OpenAlex.")
    parser.add_argument("--context-jsonl", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--researchagent-code-dir", type=Path, default=Path("/home/zb/code/ResearchAgent/code"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--iterations", type=int, default=int(os.getenv("AGENT_ITERATIONS", "3")))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    run_local_contexts(
        context_jsonl=args.context_jsonl,
        results_path=args.results_path,
        pipeline_factory=lambda: build_researchagent_pipeline(
            researchagent_code_dir=args.researchagent_code_dir,
            model_name=args.model_name,
            iterations=args.iterations,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
