import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import scripts.run_ai_researcher_local_idea_agent as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_local_ai_input(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / "ai_researcher"
    instance_path = input_dir / "benchmark/final/bus_sched/bus_evo_nn.json"
    metaprompt_path = input_dir / "benchmark/process/dataset_candidate/bus_sched/metaprompt.py"
    paper_context_path = input_dir / "paper_contexts.json"
    markdown_path = tmp_path / "source/target.md"
    pdf_path = tmp_path / "source/target.pdf"

    instance_path.parent.mkdir(parents=True)
    metaprompt_path.parent.mkdir(parents=True)
    markdown_path.parent.mkdir(parents=True)
    markdown_path.write_text("# Target Paper\n\nMethod text.", encoding="utf-8")
    pdf_path.write_bytes(b"%PDF-1.4\n")

    instance_path.write_text(
        json.dumps(
            {
                "target": "Target Paper",
                "task2": "利用进化神经网络解决公交车辆调度问题，有哪些创新想法？",
                "source_papers": [
                    {
                        "reference": "Target Paper",
                        "usage": "Use the target paper as full-text evidence.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metaprompt_path.write_text(
        'TASK = """Metaprompt task text."""\nDATASET = ""\nBASELINE = ""\nCOMPARISON = ""\nEVALUATION = ""\nREF = ""\n',
        encoding="utf-8",
    )
    paper_context_path.write_text(
        json.dumps(
            [
                {
                    "role": "target",
                    "title": "Target Paper",
                    "abstract": "Target abstract.",
                    "local_pdf": str(pdf_path),
                    "local_markdown": str(markdown_path),
                    "context": "Full-text context.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return input_dir, markdown_path, pdf_path


def test_loads_local_ai_researcher_input_and_copies_papers_to_workspace(tmp_path):
    input_dir, markdown_path, pdf_path = _write_local_ai_input(tmp_path)

    run_input = runner.load_local_run_input(
        input_dir=input_dir,
        category="bus_sched",
        instance_id="bus_evo_nn",
        task_level="task2",
    )
    prepared_papers = runner.prepare_local_paper_workspace(
        paper_contexts=run_input.paper_contexts,
        local_root=tmp_path / "run",
        workplace_name="workplace",
    )

    assert run_input.task == "Metaprompt task text."
    assert "Target Paper" in run_input.references
    assert prepared_papers[0].docker_markdown_path == "/workplace/papers/000_target_target_paper.md"
    assert prepared_papers[0].work_markdown_path.read_text(encoding="utf-8") == markdown_path.read_text(
        encoding="utf-8"
    )
    assert prepared_papers[0].work_pdf_path.read_bytes() == pdf_path.read_bytes()


def test_prepares_workspace_resolves_relative_paper_paths_from_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    input_dir, markdown_path, pdf_path = _write_local_ai_input(tmp_path)
    context_path = input_dir / "paper_contexts.json"
    contexts = json.loads(context_path.read_text(encoding="utf-8"))
    contexts[0]["local_markdown"] = str(markdown_path.relative_to(tmp_path))
    contexts[0]["local_pdf"] = str(pdf_path.relative_to(tmp_path))
    context_path.write_text(json.dumps(contexts), encoding="utf-8")

    run_input = runner.load_local_run_input(
        input_dir=input_dir,
        category="bus_sched",
        instance_id="bus_evo_nn",
        task_level="task2",
    )
    prepared_papers = runner.prepare_local_paper_workspace(
        paper_contexts=run_input.paper_contexts,
        local_root=tmp_path / "run",
        workplace_name="workplace",
    )

    assert prepared_papers[0].source_markdown_path == markdown_path
    assert prepared_papers[0].source_pdf_path == pdf_path


def test_builds_local_idea_query_without_github_or_code_resources(tmp_path):
    input_dir, _, _ = _write_local_ai_input(tmp_path)
    run_input = runner.load_local_run_input(
        input_dir=input_dir,
        category="bus_sched",
        instance_id="bus_evo_nn",
        task_level="task2",
    )
    prepared_papers = runner.prepare_local_paper_workspace(
        paper_contexts=run_input.paper_contexts,
        local_root=tmp_path / "run",
        workplace_name="workplace",
    )

    query = runner.build_local_idea_query(run_input=run_input, prepared_papers=prepared_papers)

    assert "Metaprompt task text." in query
    assert "/workplace/papers/000_target_target_paper.md" in query
    assert "open_local_file" in query
    assert "github" not in query.lower()
    assert "codebase" not in query.lower()
    assert "repository" not in query.lower()


def test_generate_ideas_with_agent_runs_five_ideas_and_selection():
    calls = []

    async def fake_agent_runner(messages, context_variables, iter_times=None):
        calls.append({"iter_times": iter_times, "messages": list(messages)})
        if iter_times == "select":
            content = "selected refined idea"
        else:
            content = f"idea {len([call for call in calls if call['iter_times'] != 'select'])}"
        return messages + [{"role": "assistant", "content": content}], dict(context_variables)

    result = asyncio.run(
        runner.generate_ideas_with_agent(
            agent_runner=fake_agent_runner,
            initial_query="Initial local-paper idea query.",
            idea_count=5,
            context_variables={"working_dir": "workplace"},
        )
    )

    assert result.ideas == ["idea 1", "idea 2", "idea 3", "idea 4", "idea 5"]
    assert result.selected_idea == "selected refined idea"
    assert [call["iter_times"] for call in calls] == [None, 1, 2, 3, 4, "select"]
    assert "You have generated 5 innovative ideas" in calls[-1]["messages"][0]["content"]


def test_writes_local_idea_agent_outputs(tmp_path):
    result = runner.IdeaAgentRunResult(
        ideas=["idea 1", "idea 2"],
        selected_idea="selected idea",
        initial_query="query",
    )
    output = runner.write_idea_outputs(
        output_dir=tmp_path / "outputs",
        category="bus_sched",
        instance_id="bus_evo_nn",
        result=result,
        prepared_papers=[
            runner.PreparedLocalPaper(
                role="target",
                title="Target Paper",
                abstract="Target abstract.",
                context="Context",
                source_markdown_path=tmp_path / "source.md",
                source_pdf_path=tmp_path / "source.pdf",
                work_markdown_path=tmp_path / "work.md",
                work_pdf_path=tmp_path / "work.pdf",
                docker_markdown_path="/workplace/papers/target.md",
                docker_pdf_path="/workplace/papers/target.pdf",
            )
        ],
    )

    payload = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert payload["selected_idea"] == "selected idea"
    assert payload["papers"][0]["docker_markdown_path"] == "/workplace/papers/target.md"
    assert "## Final Selected Idea" in output.markdown_path.read_text(encoding="utf-8")


def test_build_agent_runner_uses_original_idea_agent_without_interactive_flow_cache(monkeypatch, tmp_path):
    ai_root = tmp_path / "AI-Researcher"
    (ai_root / "research_agent").mkdir(parents=True)
    calls = {}

    class FakeResponse:
        messages = [{"role": "assistant", "content": "generated idea"}]
        context_variables = {"notes": []}

    class FakeMetaChain:
        def __init__(self, log_path):
            calls["log_path"] = log_path

        async def run_async(self, agent, messages, context_variables, debug):
            calls["run_async"] = {
                "agent": agent,
                "messages": messages,
                "context_variables": context_variables,
                "debug": debug,
            }
            return FakeResponse()

    class FakeBrowser:
        def __init__(self, **kwargs):
            calls["browser_kwargs"] = kwargs

    def fake_get_idea_agent(**kwargs):
        calls["idea_agent_kwargs"] = kwargs
        return {"agent": "original idea agent"}

    research_agent_module = types.ModuleType("research_agent")
    inno_module = types.ModuleType("research_agent.inno")
    inno_module.MetaChain = FakeMetaChain
    idea_agent_module = types.ModuleType("research_agent.inno.agents.inno_agent.idea_agent")
    idea_agent_module.get_idea_agent = fake_get_idea_agent
    markdown_browser_module = types.ModuleType("research_agent.inno.environment.markdown_browser")
    markdown_browser_module.RequestsMarkdownBrowser = FakeBrowser

    monkeypatch.setitem(sys.modules, "research_agent", research_agent_module)
    monkeypatch.setitem(sys.modules, "research_agent.inno", inno_module)
    monkeypatch.setitem(sys.modules, "research_agent.inno.agents", types.ModuleType("research_agent.inno.agents"))
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.agents.inno_agent",
        types.ModuleType("research_agent.inno.agents.inno_agent"),
    )
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.agents.inno_agent.idea_agent",
        idea_agent_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.environment",
        types.ModuleType("research_agent.inno.environment"),
    )
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.environment.markdown_browser",
        markdown_browser_module,
    )

    agent_runner = runner.build_ai_researcher_agent_runner(
        ai_researcher_root=ai_root,
        cache_dir=tmp_path / "cache",
        local_root=tmp_path / "run",
        workplace_name="workplace",
        model="ollama_chat/qwen3.6:35b",
        viewport_size=4096,
        stream=False,
    )
    messages, context_variables = asyncio.run(
        agent_runner([{"role": "user", "content": "query"}], {"notes": []}, None)
    )

    assert messages[-1]["content"] == "generated idea"
    assert context_variables == {"notes": []}
    assert calls["idea_agent_kwargs"]["code_env"] is None
    assert calls["browser_kwargs"]["workplace_name"] == "workplace"
    assert "research_agent.inno.workflow.flowcache" not in sys.modules


def test_parse_text_tool_call_only_accepts_allowed_tool_json():
    parsed = runner.parse_text_tool_call(
        '{"name": "open_local_file", "arguments": {"path": "/workplace/papers/a.md"}}',
        allowed_tool_names={"open_local_file"},
    )

    assert parsed == runner.TextToolCall(
        name="open_local_file",
        arguments={"path": "/workplace/papers/a.md"},
    )
    assert (
        runner.parse_text_tool_call(
            "### Proposed Method\nnot a tool call",
            allowed_tool_names={"open_local_file"},
        )
        is None
    )
    assert (
        runner.parse_text_tool_call(
            '{"name": "unknown_tool", "arguments": {}}',
            allowed_tool_names={"open_local_file"},
        )
        is None
    )


def test_agent_runner_falls_back_to_text_tool_call_when_litellm_does_not_emit_tool_calls(
    monkeypatch,
    tmp_path,
):
    ai_root = tmp_path / "AI-Researcher"
    (ai_root / "research_agent").mkdir(parents=True)
    calls = {"tool_paths": [], "model_messages": []}

    def fake_open_local_file(path):
        calls["tool_paths"].append(path)
        return "FILE CONTENT FROM TOOL"

    class FakeAgent:
        functions = [fake_open_local_file]
        name = "Paper Survey Agent"

    class FakeResponse:
        def __init__(self, content):
            self.messages = [{"role": "assistant", "content": content}]
            self.context_variables = {"notes": []}

    class FakeMetaChain:
        def __init__(self, log_path):
            self.run_count = 0

        async def run_async(self, agent, messages, context_variables, debug):
            self.run_count += 1
            calls["model_messages"].append(list(messages))
            if self.run_count == 1:
                return FakeResponse(
                    json.dumps(
                        {
                            "name": "fake_open_local_file",
                            "arguments": {"path": "/workplace/papers/a.md"},
                        }
                    )
                )
            return FakeResponse("final idea after reading paper")

    class FakeBrowser:
        def __init__(self, **kwargs):
            pass

    def fake_get_idea_agent(**kwargs):
        return FakeAgent()

    _install_fake_ai_researcher_modules(
        monkeypatch,
        meta_chain_cls=FakeMetaChain,
        browser_cls=FakeBrowser,
        get_idea_agent=fake_get_idea_agent,
    )

    agent_runner = runner.build_ai_researcher_agent_runner(
        ai_researcher_root=ai_root,
        cache_dir=tmp_path / "cache",
        local_root=tmp_path / "run",
        workplace_name="workplace",
        model="ollama_chat/qwen3.6:35b",
        viewport_size=4096,
        stream=False,
    )
    messages, _ = asyncio.run(
        agent_runner([{"role": "user", "content": "query"}], {"notes": []}, None)
    )

    assert calls["tool_paths"] == ["/workplace/papers/a.md"]
    assert "FILE CONTENT FROM TOOL" in calls["model_messages"][1][-1]["content"]
    assert messages[-1]["content"] == "final idea after reading paper"


def test_streaming_agent_runner_writes_chunks_as_they_arrive():
    writes = []

    class FakeAgent:
        name = "Paper Survey Agent"
        model = "ollama_chat/qwen3.6:35b"
        functions = []
        tool_choice = "auto"
        examples = None

        def instructions(self, context_variables):
            return "system prompt"

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        assert kwargs["messages"][0]["role"] == "system"
        return [
            runner.make_fake_stream_chunk("hello"),
            runner.make_fake_stream_chunk(" world"),
        ]

    agent_runner = runner.build_streaming_litellm_agent_runner(
        agent=FakeAgent(),
        completion_func=fake_completion,
        stream_writer=writes.append,
        api_base="http://localhost:11435",
    )

    messages, context = asyncio.run(
        agent_runner([{"role": "user", "content": "query"}], {"notes": []}, None)
    )

    assert "".join(writes) == "hello world"
    assert messages[-1]["content"] == "hello world"
    assert context == {"notes": []}


def test_streaming_agent_runner_executes_json_text_tool_call_before_final_answer():
    writes = []
    calls = {"tool_paths": [], "completion_messages": []}

    def fake_open_local_file(path):
        calls["tool_paths"].append(path)
        return "FILE CONTENT FROM STREAM TOOL"

    class FakeAgent:
        name = "Paper Survey Agent"
        model = "ollama_chat/qwen3.6:35b"
        functions = [fake_open_local_file]
        tool_choice = "auto"
        examples = None

        def instructions(self, context_variables):
            return "system prompt"

    def fake_completion(**kwargs):
        calls["completion_messages"].append(kwargs["messages"])
        if len(calls["completion_messages"]) == 1:
            return [
                runner.make_fake_stream_chunk(
                    json.dumps(
                        {
                            "name": "fake_open_local_file",
                            "arguments": {"path": "/workplace/papers/a.md"},
                        }
                    )
                )
            ]
        return [runner.make_fake_stream_chunk("final idea")]

    agent_runner = runner.build_streaming_litellm_agent_runner(
        agent=FakeAgent(),
        completion_func=fake_completion,
        stream_writer=writes.append,
        api_base="http://localhost:11435",
    )
    messages, _ = asyncio.run(
        agent_runner([{"role": "user", "content": "query"}], {"notes": []}, None)
    )

    assert calls["tool_paths"] == ["/workplace/papers/a.md"]
    assert "FILE CONTENT FROM STREAM TOOL" in calls["completion_messages"][1][-1]["content"]
    assert messages[-1]["content"] == "final idea"
    assert "final idea" in "".join(writes)


def test_streaming_agent_runner_uses_native_ollama_provider_and_skips_none_content():
    writes = []
    calls = []

    class FakeAgent:
        name = "Paper Survey Agent"
        model = "ollama_chat/qwen3.6:35b"
        functions = []
        tool_choice = "auto"
        examples = None

        def instructions(self, context_variables):
            return "system prompt"

    def fake_ollama_stream(**kwargs):
        calls.append(kwargs)
        return [
            runner.make_fake_stream_chunk("hello"),
            {"choices": [{"delta": {"content": None}}]},
            {"choices": [{"delta": {"thinking": " thinking"}}]},
            runner.make_fake_stream_chunk(" world"),
        ]

    agent_runner = runner.build_streaming_litellm_agent_runner(
        agent=FakeAgent(),
        stream_writer=writes.append,
        api_base="http://localhost:11435",
        ollama_chat_stream_func=fake_ollama_stream,
    )

    messages, _ = asyncio.run(
        agent_runner([{"role": "user", "content": "query"}], {"notes": []}, None)
    )

    assert calls[0]["model"] == "ollama_chat/qwen3.6:35b"
    assert messages[-1]["content"] == "hello world"
    assert "".join(writes) == "hello thinking world"


def _install_fake_ai_researcher_modules(
    monkeypatch,
    *,
    meta_chain_cls,
    browser_cls,
    get_idea_agent,
):
    research_agent_module = types.ModuleType("research_agent")
    inno_module = types.ModuleType("research_agent.inno")
    inno_module.MetaChain = meta_chain_cls
    idea_agent_module = types.ModuleType("research_agent.inno.agents.inno_agent.idea_agent")
    idea_agent_module.get_idea_agent = get_idea_agent
    markdown_browser_module = types.ModuleType("research_agent.inno.environment.markdown_browser")
    markdown_browser_module.RequestsMarkdownBrowser = browser_cls

    monkeypatch.setitem(sys.modules, "research_agent", research_agent_module)
    monkeypatch.setitem(sys.modules, "research_agent.inno", inno_module)
    monkeypatch.setitem(sys.modules, "research_agent.inno.agents", types.ModuleType("research_agent.inno.agents"))
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.agents.inno_agent",
        types.ModuleType("research_agent.inno.agents.inno_agent"),
    )
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.agents.inno_agent.idea_agent",
        idea_agent_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.environment",
        types.ModuleType("research_agent.inno.environment"),
    )
    monkeypatch.setitem(
        sys.modules,
        "research_agent.inno.environment.markdown_browser",
        markdown_browser_module,
    )


def test_local_idea_agent_script_entrypoint_can_show_help_from_repo_root():
    result = subprocess.run(
        ["python3", "scripts/run_ai_researcher_local_idea_agent.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run AI-Researcher original idea agent on local papers" in result.stdout
