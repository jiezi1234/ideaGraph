from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import opendataloader_pdf
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import Settings


LOGGER = logging.getLogger(__name__)

ABSTRACT_HEADING_RE = re.compile(r"(?im)^\s*#{0,6}\s*(?:abstract|摘要)\s*$")
INLINE_ABSTRACT_RE = re.compile(
    r"(?is)\bAbstract\s*[—\-:]\s*(.+?)(?=\n\s*(?:Index\s+Terms|Keywords|关键词)\s*[—\-:]|\n\s*(?:I\.|1\.|#)\s*INTRODUCTION|\Z)"
)
REFERENCES_HEADING_RE = re.compile(
    r"(?im)^\s*#{0,6}\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography|参考文献|引用文献)\s*$"
)
MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+.+?$")
REFERENCE_START_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\[(?P<bracket>\d+)\]|(?P<number>\d{1,3})[.)])\s+")
URL_RE = re.compile(r"https?://\S+|doi:\s*\S+", flags=re.IGNORECASE)
PLAIN_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:(?:[IVXLC]+|\d+(?:\.\d+)*)[.)]?\s+)?(?P<title>[A-Z][A-Za-z0-9 ,&()/-]{2,})\s*$"
)
DEFAULT_PAPER_CONTEXT_MAX_CHARS = 7000
SECTION_CONTEXT_MAX_CHARS = 1800
REFERENCE_USAGE_MAX_CHARS = 2800


@dataclass(frozen=True)
class Neo4jConnectionConfig:
    uri: str
    username: str
    password: str
    database: str
    timeout_seconds: float = 3.0


@dataclass(frozen=True)
class PreparedPaper:
    title: str
    abstract: str
    pdf_path: Path
    markdown_path: Path
    reference_titles: list[str]
    paper_id: str = ""
    full_text_context: str = ""


def extract_abstract_from_markdown(markdown: str) -> str:
    inline_match = INLINE_ABSTRACT_RE.search(markdown)
    if inline_match:
        return _normalize_text(inline_match.group(1))

    match = ABSTRACT_HEADING_RE.search(markdown)
    if match:
        start = match.end()
        next_heading = MARKDOWN_HEADING_RE.search(markdown, start)
        end = next_heading.start() if next_heading else len(markdown)
        abstract = markdown[start:end]
        return _normalize_text(abstract)

    paragraphs = [
        _normalize_text(part)
        for part in re.split(r"\n\s*\n", markdown)
        if _normalize_text(part) and not part.lstrip().startswith("#")
    ]
    return paragraphs[0] if paragraphs else ""


def extract_reference_titles(markdown: str, max_references: int = 20) -> list[str]:
    match = REFERENCES_HEADING_RE.search(markdown)
    if not match:
        return []

    reference_block = markdown[match.end() :]
    entries = _split_reference_entries(reference_block)
    titles = []
    seen = set()
    for entry in entries:
        title = _reference_entry_to_title(entry)
        normalized_key = title.lower()
        if title and normalized_key not in seen:
            titles.append(title)
            seen.add(normalized_key)
        if len(titles) >= max_references:
            break
    return titles


def build_ai_researcher_instance(
    *,
    target_paper: PreparedPaper,
    related_papers: Sequence[PreparedPaper],
    question: str,
    instance_id: str,
    max_references: int,
) -> dict:
    source_papers = []
    rank = 1
    for paper in related_papers:
        source_papers.append(
            {
                "reference": paper.title,
                "rank": rank,
                "type": ["local_reference"],
                "justification": "This local paper was provided as visible evidence for the same research question.",
                "usage": _build_local_paper_usage(paper),
            }
        )
        rank += 1

    for title in target_paper.reference_titles:
        if len(source_papers) >= max_references:
            break
        source_papers.append(
            {
                "reference": title,
                "rank": rank,
                "type": ["bibliographic_reference"],
                "justification": "This paper appears in the bibliography of the local target paper.",
                "usage": "Use this cited paper as background evidence from the target paper bibliography.",
            }
        )
        rank += 1

    if not source_papers:
        source_papers.append(
            {
                "reference": target_paper.title,
                "rank": 1,
                "type": ["local_target"],
                "justification": "No related local papers or bibliography entries were extracted, so the local target paper is used as primary evidence.",
                "usage": _build_local_paper_usage(target_paper, role="target"),
            }
        )

    return {
        "target": target_paper.title,
        "instance_id": instance_id,
        "url": "",
        "local_pdf": str(target_paper.pdf_path),
        "local_markdown": str(target_paper.markdown_path),
        "source_papers": source_papers,
        "local_paper_contexts": build_ai_researcher_paper_contexts(target_paper, related_papers),
        "authors": [],
        "year": None,
        "abstract": target_paper.abstract,
        "citation": None,
        "task1": f"Develop innovative research methods for this question: {question}",
        "task2": question,
    }


def build_researchagent_context(
    *,
    target_paper: PreparedPaper,
    related_papers: Sequence[PreparedPaper],
    entities: Sequence[str],
    max_references: int,
) -> dict:
    references = [
        {
            "title": paper.title,
            "abstract": paper.abstract,
            "corpusId": _local_corpus_id(paper.title),
        }
        for paper in related_papers
    ]
    for title in target_paper.reference_titles:
        if len(references) >= max_references:
            break
        references.append(
            {
                "title": title,
                "abstract": "",
                "corpusId": _local_corpus_id(title),
            }
        )

    return {
        "paper": {
            "title": target_paper.title,
            "abstract": target_paper.abstract,
            "corpusId": _local_corpus_id(target_paper.title),
        },
        "references": references,
        "entities": list(dict.fromkeys(entity.strip() for entity in entities if entity.strip())),
    }


def build_ai_researcher_paper_contexts(
    target_paper: PreparedPaper,
    related_papers: Sequence[PreparedPaper],
) -> list[dict[str, str]]:
    papers = [("target", target_paper)] + [("related", paper) for paper in related_papers]
    return [
        {
            "role": role,
            "title": paper.title,
            "abstract": paper.abstract,
            "local_pdf": str(paper.pdf_path),
            "local_markdown": str(paper.markdown_path),
            "context": paper.full_text_context or _fallback_paper_context(paper),
        }
        for role, paper in papers
    ]


def build_paper_full_text_context(
    *,
    title: str,
    abstract: str,
    markdown: str,
    max_chars: int = DEFAULT_PAPER_CONTEXT_MAX_CHARS,
) -> str:
    body = _strip_references_from_markdown(markdown)
    selected_sections = _select_representative_sections(body, document_title=title)
    parts = [f"Title: {title}"]
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if selected_sections:
        for label, section_text in selected_sections:
            parts.append(f"{label}: {_truncate_text(section_text, SECTION_CONTEXT_MAX_CHARS)}")
    else:
        # 没有可靠章节标题时，保留去参考文献后的正文前段，避免只剩标题和摘要。
        fallback = _normalize_text(body)
        if fallback:
            parts.append(f"Full text excerpt: {_truncate_text(fallback, max_chars)}")
    return _truncate_text("\n\n".join(part for part in parts if part.strip()), max_chars)


def fetch_paper_entities_from_neo4j(
    paper_id: str,
    *,
    config: Neo4jConnectionConfig | None = None,
    driver_factory=GraphDatabase.driver,
) -> list[str]:
    if not paper_id:
        return []
    if config is None:
        settings = Settings()
        config = Neo4jConnectionConfig(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )

    driver = driver_factory(
        config.uri,
        auth=(config.username, config.password),
        connection_timeout=config.timeout_seconds,
        max_transaction_retry_time=0,
    )
    try:
        records, _, _ = driver.execute_query(
            """
            MATCH (:Paper {id: $paper_id})-[:CONTAINS]->(e:Entity)
            RETURN e.id AS entity_id, e.entity_type AS entity_type
            ORDER BY coalesce(e.entity_type, ''), e.id
            """,
            parameters_={"paper_id": paper_id},
            database_=config.database,
        )
    finally:
        driver.close()

    entities = []
    seen = set()
    for record in records:
        entity_id = str(record.get("entity_id") or "").strip()
        if entity_id and entity_id not in seen:
            entities.append(entity_id)
            seen.add(entity_id)
    return entities


def resolve_entities_for_context(
    *,
    question: str,
    papers: Sequence[PreparedPaper],
    explicit_entities: Sequence[str],
    neo4j_entities: Sequence[str],
) -> list[str]:
    graph_entities = list(dict.fromkeys(entity.strip() for entity in neo4j_entities if entity.strip()))
    if graph_entities:
        return graph_entities
    return infer_entities(question, papers, explicit_entities)


def write_baseline_inputs(
    *,
    output_dir: Path,
    category: str,
    instance_id: str,
    question: str,
    target_paper: PreparedPaper,
    related_papers: Sequence[PreparedPaper],
    entities: Sequence[str],
    max_references: int,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    ai_root = output_dir / "ai_researcher"
    ai_instance_path = ai_root / "benchmark" / "final" / category / f"{instance_id}.json"
    ai_metaprompt_path = (
        ai_root / "benchmark" / "process" / "dataset_candidate" / category / "metaprompt.py"
    )
    ai_context_path = ai_root / "paper_contexts.json"
    researchagent_root = output_dir / "researchagent"
    researchagent_context_path = researchagent_root / "local_contexts.jsonl"
    manifest_path = output_dir / "manifest.json"

    ai_instance_path.parent.mkdir(parents=True, exist_ok=True)
    ai_metaprompt_path.parent.mkdir(parents=True, exist_ok=True)
    researchagent_context_path.parent.mkdir(parents=True, exist_ok=True)

    ai_instance = build_ai_researcher_instance(
        target_paper=target_paper,
        related_papers=related_papers,
        question=question,
        instance_id=instance_id,
        max_references=max_references,
    )
    _write_json(ai_instance_path, ai_instance)
    _write_json(ai_context_path, build_ai_researcher_paper_contexts(target_paper, related_papers))
    ai_metaprompt_path.write_text(
        build_ai_researcher_metaprompt(
            question=question,
            target_paper=target_paper,
            related_papers=related_papers,
            entities=entities,
        ),
        encoding="utf-8",
    )

    researchagent_context = build_researchagent_context(
        target_paper=target_paper,
        related_papers=related_papers,
        entities=entities,
        max_references=max_references,
    )
    researchagent_context_path.write_text(
        json.dumps(researchagent_context, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "target_paper": target_paper.title,
        "related_local_paper_count": str(len(related_papers)),
        "target_reference_count": str(len(target_paper.reference_titles)),
        "ai_researcher_instance_path": str(ai_instance_path),
        "ai_researcher_metaprompt_path": str(ai_metaprompt_path),
        "ai_researcher_context_path": str(ai_context_path),
        "researchagent_context_path": str(researchagent_context_path),
        "manifest_path": str(manifest_path),
    }
    manifest = {
        "target_paper": target_paper.title,
        "question": question,
        "category": category,
        "instance_id": instance_id,
        "entities": list(entities),
        "outputs": summary,
        "commands": build_run_commands(
            output_dir=output_dir,
            category=category,
            instance_id=instance_id,
            researchagent_context_path=researchagent_context_path,
        ),
    }
    _write_json(manifest_path, manifest)
    return summary


def build_ai_researcher_metaprompt(
    *,
    question: str,
    target_paper: PreparedPaper,
    related_papers: Sequence[PreparedPaper],
    entities: Sequence[str],
) -> str:
    related_titles = "\n".join(f"- {paper.title}" for paper in related_papers) or "- None provided"
    entity_text = ", ".join(entities) if entities else "bus scheduling, evolutionary neural networks"
    local_context = _format_ai_researcher_local_context(target_paper, related_papers)
    # AI-Researcher 的轻量入口只读取 TASK；其余字段保留给完整入口和人工检查。
    return (
        f'TASK = r"""\n{_safe_triple_quoted(question)}\n\n'
        f"Target local paper: {_safe_triple_quoted(target_paper.title)}\n"
        f"Focus entities: {_safe_triple_quoted(entity_text)}\n"
        "No code resources are required for this no-experiment idea-generation run. "
        "Use the local full-text paper contexts below as the substitute for AI-Researcher's "
        "paper-reading stage before generating ideas.\n\n"
        f"Local full-text paper contexts:\n{_safe_triple_quoted(local_context)}\n"
        '"""\n\n'
        'DATASET = r"""\n'
        "Use public-transit scheduling, vehicle blocking, dispatching, headway control, AVL/APC, GTFS, "
        "and electric-bus scheduling data when available.\n"
        '"""\n\n'
        'BASELINE = r"""\n'
        "Compare against mixed-integer programming, column generation, genetic algorithms, tabu search, "
        "reinforcement learning, graph neural policies, and conventional dispatching heuristics.\n"
        '"""\n\n'
        'COMPARISON = r"""\n'
        "Report operating cost, fleet size, deadhead distance, passenger waiting time, schedule delay, "
        "headway regularity, charging feasibility, constraint violations, robustness, and online latency.\n"
        '"""\n\n'
        'EVALUATION = r"""\n'
        "Evaluate feasibility, novelty, sensitivity to demand and travel-time uncertainty, multi-objective "
        "Pareto quality, and deployability in online bus scheduling.\n"
        '"""\n\n'
        f'REF = r"""\nVisible local related papers:\n{_safe_triple_quoted(related_titles)}\n"""\n'
    )


def build_run_commands(
    *,
    output_dir: Path,
    category: str,
    instance_id: str,
    researchagent_context_path: Path,
) -> dict[str, str]:
    output_dir = output_dir.resolve()
    idea_graph_root = Path(__file__).resolve().parents[1]
    return {
        "ai_researcher_local_idea_agent": (
            f"cd {idea_graph_root} && "
            "API_BASE_URL=http://10.112.24.198:30000 "
            "CHEEP_MODEL=ollama_chat/qwen3.6:35b "
            "/home/zb/code/AI-Researcher/.venv/bin/python "
            "scripts/run_ai_researcher_local_idea_agent.py "
            "--ai-researcher-root /home/zb/code/AI-Researcher "
            f"--input-dir {(output_dir / 'ai_researcher').resolve()} "
            f"--category {category} --instance-id {instance_id} --task-level task2"
        ),
        "ai_researcher_idea_only": (
            f"cd {output_dir / 'ai_researcher'} && "
            f"PYTHONPATH=$PWD CATEGORY={category} INSTANCE_ID={instance_id} TASK_LEVEL=task2 "
            "OLLAMA_BASE_URL=http://10.112.24.198:30000 "
            "COMPLETION_MODEL=ollama_chat/qwen3.6:35b "
            "/home/zb/code/AI-Researcher/.venv/bin/python /home/zb/code/AI-Researcher/run_idea_only.py"
        ),
        "researchagent_local_context": (
            f"cd {idea_graph_root} && "
            "PYTHONPATH=/home/zb/code/ResearchAgent/code "
            "/home/zb/code/ResearchAgent/.venv/bin/python "
            "scripts/run_researchagent_local_context.py "
            f"--context-jsonl {researchagent_context_path.resolve()} "
            f"--results-path {(output_dir / 'researchagent' / 'ideas.jsonl').resolve()} "
            "--researchagent-code-dir /home/zb/code/ResearchAgent/code "
            "--model-name qwen3.6:35b --iterations 3"
        ),
    }


def prepare_local_papers(
    *,
    paper_dirs: Sequence[Path],
    output_dir: Path,
    max_references: int,
) -> list[PreparedPaper]:
    converted_root = output_dir / "converted_markdown"
    prepared = []
    for paper_dir in paper_dirs:
        prepared.append(
            prepare_local_paper(
                paper_dir=paper_dir,
                converted_root=converted_root,
                max_references=max_references,
            )
        )
    return prepared


def prepare_local_paper(
    *,
    paper_dir: Path,
    converted_root: Path,
    max_references: int,
) -> PreparedPaper:
    paper_dir = paper_dir.expanduser().resolve()
    pdf_path = find_local_pdf(paper_dir)
    paper_output_dir = converted_root / _local_paper_output_slug(paper_dir)
    markdown_path = find_existing_markdown(paper_output_dir)
    if markdown_path is None:
        markdown_path = convert_pdf_to_markdown(pdf_path, paper_output_dir)

    markdown = markdown_path.read_text(encoding="utf-8")
    title = extract_title_from_markdown(markdown) or paper_dir.name
    abstract = extract_abstract_from_markdown(markdown)
    reference_titles = extract_reference_titles(markdown, max_references=max_references)
    full_text_context = build_paper_full_text_context(
        title=title,
        abstract=abstract,
        markdown=markdown,
    )
    return PreparedPaper(
        title=title,
        abstract=abstract,
        pdf_path=pdf_path,
        markdown_path=markdown_path,
        reference_titles=reference_titles,
        paper_id=paper_dir.name,
        full_text_context=full_text_context,
    )


def find_local_pdf(paper_dir: Path) -> Path:
    if not paper_dir.exists() or not paper_dir.is_dir():
        raise FileNotFoundError(f"paper folder not found: {paper_dir}")
    direct_pdfs = sorted(path for path in paper_dir.glob("*.pdf") if path.is_file())
    if len(direct_pdfs) == 1:
        return direct_pdfs[0]
    if len(direct_pdfs) > 1:
        raise ValueError(f"multiple pdf files found in {paper_dir}: {direct_pdfs}")
    recursive_pdfs = sorted(path for path in paper_dir.rglob("*.pdf") if path.is_file())
    if len(recursive_pdfs) == 1:
        return recursive_pdfs[0]
    if recursive_pdfs:
        raise ValueError(f"multiple pdf files found in {paper_dir}: {recursive_pdfs}")
    raise FileNotFoundError(f"no pdf file found in {paper_dir}")


def find_existing_markdown(output_dir: Path) -> Path | None:
    markdown_files = sorted(output_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in markdown_files:
        try:
            if path.read_text(encoding="utf-8").strip():
                return path
        except OSError:
            continue
    return None


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("pdf_convert:start pdf=%s output_dir=%s", pdf_path, output_dir)
    opendataloader_pdf.convert(
        input_path=str(pdf_path),
        output_dir=str(output_dir),
        format="markdown,json",
        image_output="off",
        quiet=True,
    )
    markdown_path = find_existing_markdown(output_dir)
    if markdown_path is None:
        raise FileNotFoundError(f"no markdown output found in {output_dir}")
    LOGGER.info("pdf_convert:done pdf=%s markdown=%s", pdf_path, markdown_path)
    return markdown_path


def extract_title_from_markdown(markdown: str) -> str:
    match = re.search(r"(?m)^\s*#\s+(.+?)\s*$", markdown)
    if not match:
        return ""
    title = _normalize_text(match.group(1))
    if title.lower() in {"source", "paper"}:
        return ""
    return title


def infer_entities(question: str, papers: Sequence[PreparedPaper], explicit_entities: Sequence[str]) -> list[str]:
    entities = [entity.strip() for entity in explicit_entities if entity.strip()]
    combined = " ".join([question] + [paper.title for paper in papers]).lower()
    phrase_rules = [
        ("公交", "bus vehicle scheduling"),
        ("bus", "bus vehicle scheduling"),
        ("online bus", "online bus scheduling"),
        ("调度", "vehicle dispatching"),
        ("scheduling", "vehicle dispatching"),
        ("进化", "evolutionary neural network"),
        ("evolutionary", "evolutionary neural network"),
        ("neural", "neural network"),
        ("charging", "electric bus charging"),
        ("uncertainty", "travel time uncertainty"),
    ]
    for marker, phrase in phrase_rules:
        if marker in combined:
            entities.append(phrase)
    return list(dict.fromkeys(entities))


def _build_local_paper_usage(paper: PreparedPaper, role: str = "related") -> str:
    context = paper.full_text_context or _fallback_paper_context(paper)
    prefix = (
        "Use this local target paper as primary full-text evidence"
        if role == "target"
        else "Use this local paper as related full-text evidence"
    )
    return f"{prefix}: {_truncate_text(context, REFERENCE_USAGE_MAX_CHARS)}"


def _format_ai_researcher_local_context(
    target_paper: PreparedPaper,
    related_papers: Sequence[PreparedPaper],
) -> str:
    blocks = []
    for item in build_ai_researcher_paper_contexts(target_paper, related_papers):
        role = "Target paper" if item["role"] == "target" else "Related paper"
        blocks.append(
            "\n".join(
                [
                    f"## {role}: {item['title']}",
                    f"Local PDF: {item['local_pdf']}",
                    f"Local Markdown: {item['local_markdown']}",
                    item["context"],
                ]
            )
        )
    return "\n\n".join(blocks)


def _fallback_paper_context(paper: PreparedPaper) -> str:
    parts = [f"Title: {paper.title}"]
    if paper.abstract:
        parts.append(f"Abstract: {paper.abstract}")
    else:
        parts.append("Abstract: No abstract extracted.")
    return "\n".join(parts)


def _strip_references_from_markdown(markdown: str) -> str:
    match = REFERENCES_HEADING_RE.search(markdown)
    return markdown[: match.start()] if match else markdown


def _select_representative_sections(markdown: str, *, document_title: str = "") -> list[tuple[str, str]]:
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("Problem framing", ("introduction", "background", "related work")),
        ("Problem and constraints", ("problem", "formulation", "constraint", "objective")),
        ("Method details", ("method", "methodology", "approach", "framework", "algorithm", "model")),
        ("Experiments and data", ("experiment", "evaluation", "result", "dataset", "data")),
        ("Limitations and future work", ("conclusion", "discussion", "limitation", "future")),
    ]
    normalized_document_title = _normalize_heading(document_title).lower()
    selected: dict[str, str] = {}
    for title, section_text in _iter_markdown_sections(markdown):
        if normalized_document_title and _normalize_heading(title).lower() == normalized_document_title:
            continue
        normalized_title = title.lower()
        normalized_section = _normalize_text(section_text)
        if not normalized_section:
            continue
        for label, keywords in groups:
            if label in selected:
                continue
            if any(keyword in normalized_title for keyword in keywords):
                selected[label] = normalized_section
                break
    return [(label, selected[label]) for label, _ in groups if label in selected]


def _iter_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    lines = markdown.splitlines()
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        label = _extract_heading_label(line)
        if label:
            headings.append((index, label))
    sections = []
    for pos, (line_index, title) in enumerate(headings):
        start = line_index + 1
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        sections.append((title, "\n".join(lines[start:end])))
    return sections


def _extract_heading_label(line: str) -> str:
    markdown_match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
    if markdown_match:
        return _normalize_heading(markdown_match.group(1))

    plain_match = PLAIN_SECTION_HEADING_RE.match(line)
    if not plain_match:
        return ""
    title = _normalize_heading(plain_match.group("title"))
    keyword_text = title.lower()
    if any(
        keyword in keyword_text
        for keyword in (
            "abstract",
            "introduction",
            "background",
            "related",
            "problem",
            "method",
            "approach",
            "framework",
            "algorithm",
            "experiment",
            "evaluation",
            "result",
            "discussion",
            "conclusion",
            "limitation",
            "future",
            "reference",
            "bibliography",
        )
    ):
        return title
    return ""


def _normalize_heading(value: str) -> str:
    value = re.sub(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*)[.)]?\s+", "", value, flags=re.IGNORECASE)
    return _normalize_text(value).strip(" .:-")


def _truncate_text(value: str, max_chars: int) -> str:
    value = _normalize_text(value)
    if len(value) <= max_chars:
        return value
    cutoff = value.rfind(" ", 0, max_chars)
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return value[:cutoff].rstrip() + "..."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local PDF papers as AI-Researcher and ResearchAgent baseline inputs."
    )
    parser.add_argument("--paper-dir", action="append", type=Path, dest="paper_dirs", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category", default="bus_sched")
    parser.add_argument("--instance-id", default="bus_evo_nn")
    parser.add_argument("--max-references", type=int, default=20)
    parser.add_argument("--entity", action="append", dest="entities", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    papers = prepare_local_papers(
        paper_dirs=args.paper_dirs,
        output_dir=args.output_dir,
        max_references=args.max_references,
    )
    target_paper = papers[0]
    related_papers = papers[1:]
    neo4j_entities = []
    try:
        neo4j_entities = fetch_paper_entities_from_neo4j(target_paper.paper_id or target_paper.title)
        if neo4j_entities:
            LOGGER.info(
                "neo4j_entities:loaded paper_id=%s count=%s",
                target_paper.paper_id or target_paper.title,
                len(neo4j_entities),
            )
        else:
            LOGGER.info("neo4j_entities:empty paper_id=%s", target_paper.paper_id or target_paper.title)
    except Exception as exc:
        # 图谱实体是增强输入，不能因为 Neo4j 暂不可用阻断 PDF baseline 输入生成。
        LOGGER.warning(
            "neo4j_entities:failed paper_id=%s error=%s message=%s",
            target_paper.paper_id or target_paper.title,
            type(exc).__name__,
            exc,
        )
    entities = resolve_entities_for_context(
        question=args.question,
        papers=papers,
        explicit_entities=args.entities,
        neo4j_entities=neo4j_entities,
    )
    summary = write_baseline_inputs(
        output_dir=args.output_dir,
        category=args.category,
        instance_id=args.instance_id,
        question=args.question,
        target_paper=target_paper,
        related_papers=related_papers,
        entities=entities,
        max_references=args.max_references,
    )
    LOGGER.info("prepare_local_baseline_inputs:summary %s", summary)
    return 0


def _split_reference_entries(reference_block: str) -> list[str]:
    entries = []
    current: list[str] = []
    for raw_line in reference_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if MARKDOWN_HEADING_RE.match(line) and current:
            break
        if REFERENCE_START_RE.match(line) and current:
            entries.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(" ".join(current))
    return entries


def _reference_entry_to_title(entry: str) -> str:
    cleaned = URL_RE.sub("", entry)
    cleaned = REFERENCE_START_RE.sub("", cleaned)
    cleaned = _normalize_text(cleaned)
    quoted = re.search(r"[\"“](.+?)[\"”]", cleaned)
    if quoted:
        return _clean_title(quoted.group(1))
    year_match = re.search(r"(?:\(\d{4}[a-z]?\)|\b\d{4}[a-z]?\b)\.?\s+(.+)", cleaned, flags=re.IGNORECASE)
    if year_match:
        cleaned = year_match.group(1)
    first_sentence = re.split(r"\.\s+", cleaned, maxsplit=1)[0]
    return _clean_title(first_sentence)


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,-;:")
    value = re.sub(r"\barXiv\b.*$", "", value, flags=re.IGNORECASE).strip(" .,-;:")
    return value


def _normalize_text(value: str) -> str:
    value = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", value)
    value = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_triple_quoted(value: str) -> str:
    return str(value).replace('"""', '\\"\\"\\"')


def _local_corpus_id(title: str) -> str:
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
    return f"local:{digest}"


def _local_paper_output_slug(paper_dir: Path) -> str:
    normalized = paper_dir.name.strip().replace(" ", "_")
    digest = hashlib.sha1(str(paper_dir).encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


if __name__ == "__main__":
    raise SystemExit(main())
