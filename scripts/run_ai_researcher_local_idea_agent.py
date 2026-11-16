from __future__ import annotations

import argparse
import asyncio
import inspect
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence


LOGGER = logging.getLogger(__name__)
CONTEXT_VARIABLES_PARAM = "context_variables"

AgentRunner = Callable[[list[dict[str, Any]], dict[str, Any], Any], Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]
CompletionFunc = Callable[..., Any]
StreamWriter = Callable[[str], None]


@dataclass(frozen=True)
class LocalRunInput:
    input_dir: Path
    category: str
    instance_id: str
    task_level: str
    task: str
    references: str
    source_papers: list[dict[str, Any]]
    paper_contexts: list[dict[str, Any]]
    instance: dict[str, Any]


@dataclass(frozen=True)
class PreparedLocalPaper:
    role: str
    title: str
    abstract: str
    context: str
    source_markdown_path: Path
    source_pdf_path: Path | None
    work_markdown_path: Path
    work_pdf_path: Path | None
    docker_markdown_path: str
    docker_pdf_path: str | None


@dataclass(frozen=True)
class IdeaAgentRunResult:
    ideas: list[str]
    selected_idea: str
    initial_query: str


@dataclass(frozen=True)
class IdeaOutputPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class TextToolCall:
    name: str
    arguments: dict[str, Any]


def load_local_run_input(
    *,
    input_dir: Path,
    category: str,
    instance_id: str,
    task_level: str,
) -> LocalRunInput:
    input_dir = Path(input_dir).expanduser().resolve()
    instance_path = input_dir / "benchmark" / "final" / category / f"{instance_id}.json"
    if not instance_path.exists():
        raise FileNotFoundError(f"AI-Researcher benchmark instance not found: {instance_path}")

    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    task = _load_metaprompt_task(input_dir=input_dir, category=category)
    if not task:
        task = str(instance.get(task_level) or instance.get("task2") or "")
    if not task:
        raise ValueError(f"task text is empty for task_level={task_level}")

    source_papers = list(instance.get("source_papers") or [])
    paper_contexts = _load_paper_contexts(input_dir=input_dir, instance=instance)
    return LocalRunInput(
        input_dir=input_dir,
        category=category,
        instance_id=instance_id,
        task_level=task_level,
        task=task,
        references=format_references(source_papers),
        source_papers=source_papers,
        paper_contexts=paper_contexts,
        instance=instance,
    )


def format_references(source_papers: Sequence[dict[str, Any]]) -> str:
    lines = []
    for source_paper in source_papers:
        title = str(source_paper.get("reference") or "").strip()
        usage = str(source_paper.get("usage") or "").strip()
        if title:
            lines.append(f"Title: {title}; You can use this paper in the following way: {usage}")
    if not lines:
        return "No additional reference titles were provided; use the local target paper full text."
    return "\n".join(lines)


def prepare_local_paper_workspace(
    *,
    paper_contexts: Sequence[dict[str, Any]],
    local_root: Path,
    workplace_name: str,
) -> list[PreparedLocalPaper]:
    local_root = Path(local_root).expanduser().resolve()
    papers_dir = local_root / workplace_name / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for index, context in enumerate(paper_contexts):
        role = str(context.get("role") or f"paper_{index}").strip() or f"paper_{index}"
        title = str(context.get("title") or f"paper_{index}").strip() or f"paper_{index}"
        markdown_path = _resolve_path(context.get("local_markdown"), fallback_root=local_root)
        if not markdown_path.exists():
            raise FileNotFoundError(f"local markdown not found for {title}: {markdown_path}")

        base_name = f"{index:03d}_{_slug(role)}_{_slug(title)}"
        work_markdown_path = papers_dir / f"{base_name}.md"
        shutil.copy2(markdown_path, work_markdown_path)

        source_pdf_path = _optional_path(context.get("local_pdf"), fallback_root=local_root)
        work_pdf_path = None
        if source_pdf_path is not None and source_pdf_path.exists():
            pdf_suffix = source_pdf_path.suffix if source_pdf_path.suffix else ".pdf"
            work_pdf_path = papers_dir / f"{base_name}{pdf_suffix}"
            shutil.copy2(source_pdf_path, work_pdf_path)

        # AI-Researcher 的 open_local_file 接收 docker 风格路径，再由 browser 映射回 local_root。
        docker_markdown_path = f"/{workplace_name}/papers/{work_markdown_path.name}"
        docker_pdf_path = f"/{workplace_name}/papers/{work_pdf_path.name}" if work_pdf_path else None
        prepared.append(
            PreparedLocalPaper(
                role=role,
                title=title,
                abstract=str(context.get("abstract") or ""),
                context=str(context.get("context") or ""),
                source_markdown_path=markdown_path,
                source_pdf_path=source_pdf_path if source_pdf_path and source_pdf_path.exists() else None,
                work_markdown_path=work_markdown_path,
                work_pdf_path=work_pdf_path,
                docker_markdown_path=docker_markdown_path,
                docker_pdf_path=docker_pdf_path,
            )
        )
    if not prepared:
        raise ValueError("no local paper contexts were found")
    return prepared


def build_local_idea_query(
    *,
    run_input: LocalRunInput,
    prepared_papers: Sequence[PreparedLocalPaper],
) -> str:
    paper_files = "\n".join(
        f"- {paper.role}: {paper.title}\n"
        f"  Markdown: {paper.docker_markdown_path}\n"
        f"  PDF: {paper.docker_pdf_path or 'not provided'}"
        for paper in prepared_papers
    )
    paper_summaries = "\n\n".join(
        "\n".join(
            part
            for part in [
                f"## {paper.role}: {paper.title}",
                f"Abstract: {paper.abstract}" if paper.abstract else "",
                paper.context,
            ]
            if part.strip()
        )
        for paper in prepared_papers
    )
    return f"""\
I have a task related to machine learning:
{run_input.task}

And a list of papers for your reference:
{run_input.references}

I have prepared the corresponding full-text paper files locally in the paper folder.
Open and inspect the local Markdown files with `open_local_file` before generating ideas.

Local paper files:
{paper_files}

Local paper summaries:
{paper_summaries}

Your task is to thoroughly review these local research papers and generate innovative ideas for the given task.

Requirements:
- Use only the provided local papers and their summaries as evidence.
- Identify research gaps, technical challenges, and methodological opportunities.
- Provide detailed technical solutions and complete mathematical formulations when relevant.
- Focus on research-method ideas only, ending at idea generation and selection.
"""


async def generate_ideas_with_agent(
    *,
    agent_runner: AgentRunner,
    initial_query: str,
    idea_count: int = 5,
    context_variables: dict[str, Any] | None = None,
) -> IdeaAgentRunResult:
    if idea_count < 1:
        raise ValueError("idea_count must be at least 1")

    context_variables = dict(context_variables or {})
    messages = [{"role": "user", "content": initial_query}]
    survey_messages, context_variables = await agent_runner(messages, context_variables, None)
    ideas = [_last_message_content(survey_messages)]

    for index in range(idea_count - 1):
        next_messages = list(survey_messages)
        next_messages.append(
            {
                "role": "user",
                "content": "please survey again and give me another idea based only on the local papers",
            }
        )
        survey_messages, context_variables = await agent_runner(
            next_messages,
            context_variables,
            index + 1,
        )
        ideas.append(_last_message_content(survey_messages))

    selection_prompt = f"""\
You have generated {idea_count} innovative ideas for the given task:
{_join_ideas(ideas)}

Your task is to analyze multiple existing ideas, select the most novel one, enhance the idea if any key information is missing, and finally give me the most novel idea with refined mathematical formulation and method design.
Directly output the selected refined idea report.
"""
    selection_messages, _ = await agent_runner(
        [{"role": "user", "content": selection_prompt}],
        context_variables,
        "select",
    )
    return IdeaAgentRunResult(
        ideas=ideas,
        selected_idea=_last_message_content(selection_messages),
        initial_query=initial_query,
    )


def build_ai_researcher_agent_runner(
    *,
    ai_researcher_root: Path,
    cache_dir: Path,
    local_root: Path,
    workplace_name: str,
    model: str,
    viewport_size: int,
    stream: bool = True,
) -> AgentRunner:
    ai_researcher_root = Path(ai_researcher_root).expanduser().resolve()
    research_agent_root = ai_researcher_root / "research_agent"
    if not research_agent_root.exists():
        raise FileNotFoundError(f"AI-Researcher research_agent folder not found: {research_agent_root}")
    for path in [str(ai_researcher_root), str(research_agent_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # AI-Researcher 的 PaperMemory 初始化会读取 OPENAI_API_KEY；本地 Ollama/OpenAI-compatible 服务不一定使用它。
    os.environ.setdefault("OPENAI_API_KEY", "local-paper-runner")
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = Path(local_root) / workplace_name / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    from research_agent.inno.agents.inno_agent.idea_agent import get_idea_agent
    from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser

    with _pushd(cache_dir):
        file_env = RequestsMarkdownBrowser(
            local_root=str(Path(local_root).expanduser().resolve()),
            workplace_name=workplace_name,
            viewport_size=viewport_size,
            downloads_folder=str(downloads_dir),
        )
    agent = get_idea_agent(model=model, file_env=file_env, code_env=None)

    if stream:
        return build_streaming_litellm_agent_runner(
            agent=agent,
            api_base=_resolve_ai_researcher_api_base(),
            stream_writer=_default_stream_writer,
        )

    from research_agent.inno import MetaChain

    client = MetaChain(log_path=str(cache_dir / "metachain.log"))

    async def run_agent(
        messages: list[dict[str, Any]],
        context_variables: dict[str, Any],
        iter_times: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        current_messages = list(messages)
        current_context = dict(context_variables)
        for fallback_round in range(8):
            response = await client.run_async(
                agent,
                current_messages,
                context_variables=current_context,
                debug=True,
            )
            if response.messages and response.messages[-1].get("role") == "error":
                raise RuntimeError(response.messages[-1].get("content", "AI-Researcher idea agent failed"))

            combined_messages = current_messages + list(response.messages)
            current_context = dict(response.context_variables)
            text_tool_call = _extract_last_text_tool_call(agent=agent, messages=response.messages)
            if text_tool_call is None:
                return combined_messages, current_context

            # LiteLLM 对 Ollama 多工具场景可能只返回普通 JSON 文本；这里仅在标准 tool_calls 失效后兜底。
            tool_result, current_context = execute_text_tool_call(
                tool_call=text_tool_call,
                functions=getattr(agent, "functions", []),
                context_variables=current_context,
            )
            LOGGER.info(
                "ai_researcher_local_idea_agent:text_tool_fallback round=%s tool=%s",
                fallback_round,
                text_tool_call.name,
            )
            current_messages = combined_messages + [
                build_text_tool_observation_message(text_tool_call, tool_result)
            ]

        raise RuntimeError("AI-Researcher idea agent kept returning text tool calls without a final answer")

    return run_agent


async def run_local_idea_agent(
    *,
    ai_researcher_root: Path,
    input_dir: Path,
    category: str,
    instance_id: str,
    task_level: str,
    output_dir: Path,
    workplace_name: str,
    cache_dir: Path,
    model: str,
    idea_count: int,
    viewport_size: int,
    stream: bool,
) -> IdeaOutputPaths:
    run_input = load_local_run_input(
        input_dir=input_dir,
        category=category,
        instance_id=instance_id,
        task_level=task_level,
    )
    prepared_papers = prepare_local_paper_workspace(
        paper_contexts=run_input.paper_contexts,
        local_root=output_dir,
        workplace_name=workplace_name,
    )
    initial_query = build_local_idea_query(run_input=run_input, prepared_papers=prepared_papers)
    agent_runner = build_ai_researcher_agent_runner(
        ai_researcher_root=ai_researcher_root,
        cache_dir=cache_dir,
        local_root=output_dir,
        workplace_name=workplace_name,
        model=model,
        viewport_size=viewport_size,
        stream=stream,
    )
    result = await generate_ideas_with_agent(
        agent_runner=agent_runner,
        initial_query=initial_query,
        idea_count=idea_count,
        context_variables={"working_dir": workplace_name, "notes": []},
    )
    return write_idea_outputs(
        output_dir=output_dir,
        category=category,
        instance_id=instance_id,
        result=result,
        prepared_papers=prepared_papers,
    )


def write_idea_outputs(
    *,
    output_dir: Path,
    category: str,
    instance_id: str,
    result: IdeaAgentRunResult,
    prepared_papers: Sequence[PreparedLocalPaper],
) -> IdeaOutputPaths:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"ideas_{category}_{instance_id}.json"
    markdown_path = output_dir / f"ideas_{category}_{instance_id}.md"
    payload = {
        "category": category,
        "instance_id": instance_id,
        "initial_query": result.initial_query,
        "selected_idea": result.selected_idea,
        "ideas": result.ideas,
        "papers": [_paper_to_json(paper) for paper in prepared_papers],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ideas_markdown = "\n\n---\n\n".join(
        f"### Idea {index + 1}\n\n{idea}" for index, idea in enumerate(result.ideas)
    )
    markdown_path.write_text(
        "\n".join(
            [
                "# AI-Researcher Local Paper Idea Agent Results",
                "",
                f"- **Category**: {category}",
                f"- **Instance**: {instance_id}",
                f"- **Local Papers**: {len(prepared_papers)}",
                "",
                "## Final Selected Idea",
                "",
                result.selected_idea,
                "",
                "## All Generated Ideas",
                "",
                ideas_markdown,
                "",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info("ai_researcher_local_idea_agent:outputs json=%s markdown=%s", json_path, markdown_path)
    return IdeaOutputPaths(json_path=json_path, markdown_path=markdown_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AI-Researcher original idea agent on local papers.",
    )
    parser.add_argument("--ai-researcher-root", type=Path, default=Path("/home/zb/code/AI-Researcher"))
    parser.add_argument("--input-dir", type=Path, required=True, help="Generated ai_researcher input directory.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--task-level", default="task2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workplace-name", default="workplace")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=os.getenv("CHEEP_MODEL") or os.getenv("COMPLETION_MODEL") or "ollama_chat/qwen3.6:35b",
    )
    parser.add_argument("--idea-count", type=int, default=5)
    parser.add_argument("--viewport-size", type=int, default=8192)
    parser.add_argument("--no-stream", action="store_true", help="Use the original non-streaming MetaChain runner.")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    output_dir = args.output_dir or (args.input_dir / "outputs" / "local_idea_agent")
    cache_dir = args.cache_dir or (output_dir / "cache")
    output_paths = asyncio.run(
        run_local_idea_agent(
            ai_researcher_root=args.ai_researcher_root,
            input_dir=args.input_dir,
            category=args.category,
            instance_id=args.instance_id,
            task_level=args.task_level,
            output_dir=output_dir,
            workplace_name=args.workplace_name,
            cache_dir=cache_dir,
            model=args.model,
            idea_count=args.idea_count,
            viewport_size=args.viewport_size,
            stream=not args.no_stream,
        )
    )
    LOGGER.info("ai_researcher_local_idea_agent:done json=%s markdown=%s", output_paths.json_path, output_paths.markdown_path)
    return 0


def _load_metaprompt_task(*, input_dir: Path, category: str) -> str:
    metaprompt_path = input_dir / "benchmark" / "process" / "dataset_candidate" / category / "metaprompt.py"
    if not metaprompt_path.exists():
        return ""
    spec = importlib.util.spec_from_file_location(f"_local_metaprompt_{category}", metaprompt_path)
    if spec is None or spec.loader is None:
        return ""
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(getattr(module, "TASK", "") or "")


def _load_paper_contexts(*, input_dir: Path, instance: dict[str, Any]) -> list[dict[str, Any]]:
    context_path = input_dir / "paper_contexts.json"
    if context_path.exists():
        contexts = json.loads(context_path.read_text(encoding="utf-8"))
    else:
        contexts = instance.get("local_paper_contexts") or []
    if not isinstance(contexts, list):
        raise ValueError(f"paper contexts must be a list: {context_path}")
    if not contexts:
        local_markdown = instance.get("local_markdown")
        if local_markdown:
            contexts = [
                {
                    "role": "target",
                    "title": instance.get("target") or "target paper",
                    "abstract": instance.get("abstract") or "",
                    "local_pdf": instance.get("local_pdf") or "",
                    "local_markdown": local_markdown,
                    "context": instance.get("abstract") or "",
                }
            ]
    if not contexts:
        raise ValueError("no local paper contexts available; run prepare_local_baseline_inputs.py first")
    return [dict(context) for context in contexts]


def parse_text_tool_call(content: str, *, allowed_tool_names: set[str]) -> TextToolCall | None:
    raw = _strip_json_fence(content.strip())
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("function")
    if not isinstance(name, str) or name not in allowed_tool_names:
        return None
    arguments = payload.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return TextToolCall(name=name, arguments=arguments)


def build_streaming_litellm_agent_runner(
    *,
    agent: Any,
    api_base: str,
    completion_func: CompletionFunc | None = None,
    ollama_chat_stream_func: CompletionFunc | None = None,
    stream_writer: StreamWriter | None = None,
    max_fallback_rounds: int = 8,
) -> AgentRunner:
    completion_func = completion_func or _select_default_stream_completion(
        model=str(getattr(agent, "model", "")),
        ollama_chat_stream_func=ollama_chat_stream_func,
    )
    stream_writer = stream_writer or _default_stream_writer

    async def run_agent(
        messages: list[dict[str, Any]],
        context_variables: dict[str, Any],
        iter_times: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        current_messages = list(messages)
        current_context = dict(context_variables)
        for fallback_round in range(max_fallback_rounds):
            assistant_message = _run_streaming_completion_once(
                agent=agent,
                messages=current_messages,
                context_variables=current_context,
                api_base=api_base,
                completion_func=completion_func,
                stream_writer=stream_writer,
            )
            combined_messages = current_messages + [assistant_message]
            standard_tool_messages, current_context = execute_standard_tool_calls(
                assistant_message=assistant_message,
                functions=getattr(agent, "functions", []),
                context_variables=current_context,
            )
            if standard_tool_messages:
                LOGGER.info(
                    "ai_researcher_local_idea_agent:stream_tool_calls round=%s count=%s",
                    fallback_round,
                    len(standard_tool_messages),
                )
                current_messages = combined_messages + standard_tool_messages
                continue

            text_tool_call = _parse_text_tool_call_from_message(agent=agent, message=assistant_message)
            if text_tool_call is None:
                return combined_messages, current_context

            # Ollama 多工具 fallback 常把工具调用吐成 JSON 文本；流完后再执行工具并把结果喂回模型。
            tool_result, current_context = execute_text_tool_call(
                tool_call=text_tool_call,
                functions=getattr(agent, "functions", []),
                context_variables=current_context,
            )
            LOGGER.info(
                "ai_researcher_local_idea_agent:stream_text_tool_fallback round=%s tool=%s",
                fallback_round,
                text_tool_call.name,
            )
            current_messages = combined_messages + [
                build_text_tool_observation_message(text_tool_call, tool_result)
            ]

        raise RuntimeError("AI-Researcher streaming idea agent kept calling tools without a final answer")

    return run_agent


def execute_text_tool_call(
    *,
    tool_call: TextToolCall,
    functions: Sequence[Any],
    context_variables: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    function_map = {getattr(function, "__name__", ""): function for function in functions}
    function = function_map.get(tool_call.name)
    if function is None:
        return f"[Tool Call Error] Error: Tool {tool_call.name} not found.", context_variables
    try:
        raw_result = _call_tool_function(function, tool_call.arguments, context_variables=context_variables)
    except Exception as exc:
        return f"[Tool Call Error] The execution of tool {tool_call.name} failed. Error: {exc}", context_variables

    result_text, next_context = _normalize_tool_result(raw_result, context_variables=context_variables)
    return result_text, next_context


def execute_standard_tool_calls(
    *,
    assistant_message: dict[str, Any],
    functions: Sequence[Any],
    context_variables: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tool_calls = assistant_message.get("tool_calls") or []
    if not tool_calls:
        return [], context_variables

    function_map = {getattr(function, "__name__", ""): function for function in functions}
    next_context = dict(context_variables)
    tool_messages = []
    for tool_call in tool_calls:
        function_payload = tool_call.get("function") or {}
        name = function_payload.get("name")
        arguments = _decode_tool_arguments(function_payload.get("arguments") or "{}")
        tool_call_id = tool_call.get("id") or f"call_{len(tool_messages)}"
        function = function_map.get(name)
        if function is None:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": str(name or ""),
                    "content": f"[Tool Call Error] Error: Tool {name} not found.",
                }
            )
            continue
        try:
            raw_result = _call_tool_function(function, arguments, context_variables=next_context)
            result_text, next_context = _normalize_tool_result(raw_result, context_variables=next_context)
        except Exception as exc:
            result_text = f"[Tool Call Error] The execution of tool {name} failed. Error: {exc}"
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": str(name),
                "content": result_text,
            }
        )
    return tool_messages, next_context


def _run_streaming_completion_once(
    *,
    agent: Any,
    messages: Sequence[dict[str, Any]],
    context_variables: dict[str, Any],
    api_base: str,
    completion_func: CompletionFunc,
    stream_writer: StreamWriter,
) -> dict[str, Any]:
    model_messages = _build_agent_model_messages(
        agent=agent,
        messages=messages,
        context_variables=context_variables,
    )
    tools = _build_litellm_tools(getattr(agent, "functions", []))
    create_params: dict[str, Any] = {
        "model": getattr(agent, "model", None),
        "messages": _sanitize_model_messages(model_messages),
        "stream": True,
        "base_url": api_base,
    }
    if tools:
        create_params["tools"] = tools
        create_params["tool_choice"] = getattr(agent, "tool_choice", None)
        if getattr(agent, "parallel_tool_calls", False):
            create_params["parallel_tool_calls"] = getattr(agent, "parallel_tool_calls")

    LOGGER.info(
        "ai_researcher_local_idea_agent:stream_request model=%s messages=%s tools=%s",
        create_params["model"],
        len(create_params["messages"]),
        len(tools),
    )
    stream = completion_func(**create_params)
    try:
        assistant_message = _collect_streamed_assistant_message(
            stream=stream,
            stream_writer=stream_writer,
        )
    finally:
        _close_stream_if_possible(stream)
    assistant_message["sender"] = getattr(agent, "name", "Agent")
    if not assistant_message.get("content") and assistant_message.get("thinking"):
        assistant_message["content"] = str(assistant_message["thinking"])
    return assistant_message


def _build_agent_model_messages(
    *,
    agent: Any,
    messages: Sequence[dict[str, Any]],
    context_variables: dict[str, Any],
) -> list[dict[str, Any]]:
    instructions = _render_agent_template(getattr(agent, "instructions", ""), context_variables)
    history = [dict(message) for message in messages]
    examples = _render_agent_examples(getattr(agent, "examples", None), context_variables)
    return [{"role": "system", "content": instructions}] + examples + history


def _render_agent_template(template: Any, context_variables: dict[str, Any]) -> str:
    if callable(template):
        try:
            return str(template(context_variables))
        except TypeError:
            return str(template())
    return str(template or "")


def _render_agent_examples(examples: Any, context_variables: dict[str, Any]) -> list[dict[str, Any]]:
    if not examples:
        return []
    value = _render_agent_template(examples, context_variables) if callable(examples) else examples
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _build_litellm_tools(functions: Sequence[Any]) -> list[dict[str, Any]]:
    converter = _load_function_to_json()
    tools = [converter(function) for function in functions]
    for tool in tools:
        parameters = tool.get("function", {}).get("parameters", {})
        properties = parameters.get("properties") or {}
        properties.pop(CONTEXT_VARIABLES_PARAM, None)
        required = parameters.get("required") or []
        if CONTEXT_VARIABLES_PARAM in required:
            required.remove(CONTEXT_VARIABLES_PARAM)
    return tools


def _collect_streamed_assistant_message(
    *,
    stream: Iterable[Any],
    stream_writer: StreamWriter,
) -> dict[str, Any]:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for chunk in stream:
        delta = _first_choice_delta(chunk)
        if delta is None:
            continue
        thinking = _first_present_field(delta, ["reasoning_content", "thinking"])
        if isinstance(thinking, str) and thinking:
            thinking_parts.append(thinking)
            stream_writer(thinking)
        content = _field(delta, "content")
        if isinstance(content, str) and content:
            content_parts.append(content)
            stream_writer(content)
        delta_tool_calls = _field(delta, "tool_calls")
        if delta_tool_calls:
            _merge_stream_tool_calls(tool_calls, delta_tool_calls)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if thinking_parts:
        message["thinking"] = "".join(thinking_parts)
    clean_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call.get("function", {}).get("name") or tool_call.get("function", {}).get("arguments")
    ]
    if clean_tool_calls:
        message["tool_calls"] = clean_tool_calls
    return message


def _merge_stream_tool_calls(
    tool_calls: list[dict[str, Any]],
    delta_tool_calls: Iterable[Any],
) -> None:
    for delta_tool_call in delta_tool_calls:
        index = _field(delta_tool_call, "index")
        if index is None:
            index = len(tool_calls)
        index = int(index)
        while len(tool_calls) <= index:
            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        current = tool_calls[index]
        tool_call_id = _field(delta_tool_call, "id")
        if tool_call_id:
            current["id"] = str(tool_call_id)
        tool_call_type = _field(delta_tool_call, "type")
        if tool_call_type:
            current["type"] = str(tool_call_type)
        function_delta = _field(delta_tool_call, "function") or {}
        function_name = _field(function_delta, "name")
        if function_name:
            current["function"]["name"] += str(function_name)
        arguments = _field(function_delta, "arguments")
        if arguments:
            current["function"]["arguments"] += str(arguments)


def _first_choice_delta(chunk: Any) -> Any | None:
    choices = _field(chunk, "choices")
    if not choices:
        return None
    first_choice = choices[0]
    return _field(first_choice, "delta") or _field(first_choice, "message")


def _first_present_field(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        field_value = _field(value, name)
        if field_value:
            return field_value
    return None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _sanitize_model_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for message in messages:
        clean_message = dict(message)
        clean_message.pop("sender", None)
        sanitized.append(clean_message)
    return sanitized


def _parse_text_tool_call_from_message(*, agent: Any, message: dict[str, Any]) -> TextToolCall | None:
    if message.get("tool_calls"):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    allowed_tool_names = {
        getattr(function, "__name__", "")
        for function in getattr(agent, "functions", [])
        if getattr(function, "__name__", "")
    }
    if not allowed_tool_names:
        return None
    return parse_text_tool_call(content, allowed_tool_names=allowed_tool_names)


def _decode_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _call_tool_function(
    function: Any,
    arguments: dict[str, Any],
    *,
    context_variables: dict[str, Any],
) -> Any:
    call_arguments = dict(arguments)
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    if (
        signature is not None
        and CONTEXT_VARIABLES_PARAM in signature.parameters
        and CONTEXT_VARIABLES_PARAM not in call_arguments
    ):
        call_arguments[CONTEXT_VARIABLES_PARAM] = context_variables
    return function(**call_arguments)


def _normalize_tool_result(
    raw_result: Any,
    *,
    context_variables: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    next_context = dict(context_variables)
    result_context = getattr(raw_result, "context_variables", None)
    if isinstance(result_context, dict):
        next_context.update(result_context)
    if hasattr(raw_result, "value"):
        return str(raw_result.value), next_context
    return str(raw_result), next_context


def _load_litellm_completion() -> CompletionFunc:
    from litellm import completion

    return completion


def _select_default_stream_completion(
    *,
    model: str,
    ollama_chat_stream_func: CompletionFunc | None = None,
) -> CompletionFunc:
    if _is_ollama_chat_model(model):
        return ollama_chat_stream_func or _ollama_chat_completion_stream
    return _load_litellm_completion()


def _ollama_chat_completion_stream(**create_params: Any) -> Iterable[dict[str, Any]]:
    model = _ollama_model_name(str(create_params["model"]))
    payload: dict[str, Any] = {
        "model": model,
        "messages": _convert_messages_for_ollama(create_params["messages"]),
        "stream": True,
    }
    tools = create_params.get("tools")
    if tools:
        payload["tools"] = tools

    # Ollama 的原生 API 不需要 tool_choice；LiteLLM 也会移除它，否则部分版本会卡住。
    url = _ollama_chat_url(str(create_params.get("base_url") or "http://localhost:11434"))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                delta = _ollama_chunk_to_openai_delta(chunk)
                if delta:
                    yield {"choices": [{"delta": delta}]}
                if chunk.get("done"):
                    break
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama chat request failed with HTTP {exc.code}: {body}") from exc


def _ollama_chunk_to_openai_delta(chunk: dict[str, Any]) -> dict[str, Any]:
    message = chunk.get("message") or {}
    delta: dict[str, Any] = {}
    if "content" in message:
        delta["content"] = message.get("content")
    if "thinking" in message:
        delta["thinking"] = message.get("thinking")
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        delta["tool_calls"] = _convert_ollama_tool_calls_to_openai_delta(tool_calls)
    return delta


def _convert_ollama_tool_calls_to_openai_delta(tool_calls: Sequence[Any]) -> list[dict[str, Any]]:
    converted = []
    for index, tool_call in enumerate(tool_calls):
        function_payload = _field(tool_call, "function") or {}
        name = _field(function_payload, "name") or ""
        arguments = _field(function_payload, "arguments") or {}
        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False)
        converted.append(
            {
                "index": index,
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": arguments_json,
                },
            }
        )
    return converted


def _convert_messages_for_ollama(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        clean_message = dict(message)
        clean_message.pop("sender", None)
        if isinstance(clean_message.get("tool_calls"), list):
            clean_message["tool_calls"] = [
                _convert_openai_tool_call_for_ollama(tool_call)
                for tool_call in clean_message["tool_calls"]
            ]
        converted.append(clean_message)
    return converted


def _convert_openai_tool_call_for_ollama(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function") or {}
    return {
        "function": {
            "name": function_payload.get("name") or "",
            "arguments": _decode_tool_arguments(function_payload.get("arguments") or "{}"),
        }
    }


def _is_ollama_chat_model(model: str) -> bool:
    return model.startswith("ollama_chat/")


def _ollama_model_name(model: str) -> str:
    return model.removeprefix("ollama_chat/")


def _ollama_chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/api/chat"):
        return base_url
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return f"{base_url}/api/chat"


def _resolve_ai_researcher_api_base() -> str:
    if os.getenv("API_BASE_URL"):
        return str(os.getenv("API_BASE_URL"))
    try:
        from research_agent.constant import API_BASE_URL
    except Exception:
        return "http://localhost:11435"
    return str(API_BASE_URL)


def _load_function_to_json() -> Callable[[Any], dict[str, Any]]:
    try:
        from research_agent.inno.util import function_to_json
    except Exception:
        return _function_to_json_minimal
    return function_to_json


def _function_to_json_minimal(function: Any) -> dict[str, Any]:
    signature = inspect.signature(function)
    properties = {}
    required = []
    for parameter in signature.parameters.values():
        properties[parameter.name] = {"type": _json_schema_type(parameter.annotation)}
        if parameter.default is inspect._empty:
            required.append(parameter.name)
    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _json_schema_type(annotation: Any) -> str:
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }.get(annotation, "string")


def _default_stream_writer(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _close_stream_if_possible(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def make_fake_stream_chunk(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta}]}


def build_text_tool_observation_message(tool_call: TextToolCall, tool_result: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"The previous assistant response requested the local tool `{tool_call.name}` with arguments:\n"
            f"{json.dumps(tool_call.arguments, ensure_ascii=False)}\n\n"
            f"Tool result:\n{tool_result}\n\n"
            "Continue the paper review and produce the requested idea when enough evidence has been gathered."
        ),
    }


def _optional_path(value: Any, *, fallback_root: Path) -> Path | None:
    if not value:
        return None
    return _resolve_path(value, fallback_root=fallback_root)


def _resolve_path(value: Any, *, fallback_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    # prepare_local_baseline_inputs.py can emit paths relative to the repo cwd
    # when --output-dir itself is relative; keep cwd resolution before output dir.
    for root in [Path.cwd(), Path(fallback_root).expanduser().resolve()]:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _slug(value: str, max_length: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return (slug or "paper")[:max_length].strip("_") or "paper"


def _last_message_content(messages: Sequence[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    raise ValueError("agent returned no textual content")


def _extract_last_text_tool_call(*, agent: Any, messages: Sequence[dict[str, Any]]) -> TextToolCall | None:
    allowed_tool_names = {
        getattr(function, "__name__", "")
        for function in getattr(agent, "functions", [])
        if getattr(function, "__name__", "")
    }
    if not allowed_tool_names:
        return None
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        # 如果 LiteLLM 已经返回标准 tool_calls，MetaChain 原逻辑会负责执行，这里不抢。
        if message.get("tool_calls"):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return parse_text_tool_call(content, allowed_tool_names=allowed_tool_names)
    return None


def _strip_json_fence(content: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    return match.group(1).strip() if match else content


def _join_ideas(ideas: Sequence[str]) -> str:
    return "\n===================\n===================\n".join(ideas)


def _paper_to_json(paper: PreparedLocalPaper) -> dict[str, str | None]:
    return {
        "role": paper.role,
        "title": paper.title,
        "abstract": paper.abstract,
        "context": paper.context,
        "source_markdown_path": str(paper.source_markdown_path),
        "source_pdf_path": str(paper.source_pdf_path) if paper.source_pdf_path else None,
        "work_markdown_path": str(paper.work_markdown_path),
        "work_pdf_path": str(paper.work_pdf_path) if paper.work_pdf_path else None,
        "docker_markdown_path": paper.docker_markdown_path,
        "docker_pdf_path": paper.docker_pdf_path,
    }


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
