import logging
import os
from dotenv import load_dotenv

# 自动加载 .env 文件里的变量到系统环境变量中，这样底层的 http 库就可以识别到 NO_PROXY
load_dotenv()

# Windows 注册表代理桥接：当系统代理开启时，自动将代理配置注入 os.environ
# 这样 Python 的 httpx, requests, urllib3 就能正确识别 http_proxy 并完美遵循 NO_PROXY
if os.name == "nt":
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        ) as key:
            proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if proxy_enable == 1:
                proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
                if proxy_server:
                    proxy_url = f"http://{proxy_server}"
                    # 仅在环境变量未手动指定时注入，避免覆盖用户自定义设置
                    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
                        if not os.environ.get(var):
                            os.environ[var] = proxy_url
    except Exception:
        pass

# 确保 lowercase 和 uppercase 的 NO_PROXY/no_proxy 均被设置，确保内网 IP 直连
no_proxy_val = os.environ.get("NO_PROXY", "127.0.0.1,localhost,10.112.24.198,10.160.4.178")
os.environ["no_proxy"] = no_proxy_val
os.environ["NO_PROXY"] = no_proxy_val


from src.interfaces.cli.build_graph import build_parser
from src.app.bootstrap import build_mvp_workflow
from src.app.bootstrap import build_idea_chat_workflow
from src.app.bootstrap import build_accepted_paper_benchmark_workflow
from src.app.bootstrap import build_entity_extraction_preview_workflow
from src.app.bootstrap import build_local_paper_folder_workflow
from src.config.settings import Settings
from src.logging_config import configure_logging
from src.workflows.ai_idea_bench_pilot import AIIdeaBenchPilotWorkflow
from src.workflows.entity_extraction_preview import format_preview_json
from src.workflows.entity_extraction_preview import format_preview_table
from src.workflows.query_archive import archive_chat_idea_result



logger = logging.getLogger(__name__)

# Helper functions to normalize CLI text output, especially for handling escaped newlines and carriage returns.
def _normalize_cli_text(value):
    if isinstance(value, str):
        return value.replace('\\n', '\n').replace('\\r', '')
    return value

# Normalize lists of text items for CLI output.
def _normalize_cli_text_list(values):
    return [_normalize_cli_text(item) for item in values or []]


def run_build_graph(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_mvp_workflow(settings)
    summary = workflow.run(
        keyword=args.keyword,
        max_seed_papers=args.max_seed_papers,
        require_references=args.require_references,
    )
    logger.info("build_graph:summary %s", summary)


def run_build_openalex_seed_graph(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_mvp_workflow(settings)
    summary = workflow.run_openalex_seed_graph(openalex_ids=args.openalex_ids)
    logger.info("build_openalex_seed_graph:summary %s", summary)
    print(f"OpenAlex seed graph done: {summary}")


def run_build_local_paper_graph(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_local_paper_folder_workflow(settings)
    summary = workflow.run_local_paper_folders(args.paper_dirs)
    logger.info("build_local_paper_graph:summary %s", summary)
    print(f"Local paper graph done: {summary}")


def run_benchmark_accepted_ideas(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_accepted_paper_benchmark_workflow(settings)
    summary = workflow.run(
        venue=args.venue,
        query=args.query,
        max_papers=args.max_papers,
        output_dir=args.output_dir,
        decision_scope=args.decision_scope,
    )
    logger.info("benchmark_accepted_ideas:summary %s", summary)
    print(f"OpenReview paper benchmark done: {summary}")


def run_prepare_ai_idea_bench_pilot(args) -> None:
    configure_logging()
    workflow = AIIdeaBenchPilotWorkflow()
    summary = workflow.run(
        dataset_path=args.dataset_json,
        count=args.count,
        output_dir=args.output_dir,
    )
    logger.info("prepare_ai_idea_bench_pilot:summary %s", summary)
    print(f"AI Idea Bench pilot prepared: {summary}")


def run_preview_entity_extraction(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_entity_extraction_preview_workflow(settings)
    preview = workflow.run(args.markdown_path)
    if args.output == "json":
        print(format_preview_json(preview), end="")
        return
    if args.output == "table":
        print(format_preview_table(preview), end="")
        return
    raise ValueError(f"Unsupported output format: {args.output}")


def run_chat_idea(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    generation_profile = getattr(args, "generation_profile", "ideagraph")
    retrieval_mode = getattr(args, "retrieval_mode", "graph")
    workflow_kwargs = {}
    if generation_profile != "ideagraph":
        workflow_kwargs["generation_profile"] = generation_profile
    if retrieval_mode != "graph":
        workflow_kwargs["retrieval_mode"] = retrieval_mode
    workflow = build_idea_chat_workflow(settings, **workflow_kwargs)
    seed_paper_id = getattr(args, "seed_paper_id", None)
    if seed_paper_id:
        result = workflow.run(user_query=args.query, seed_paper_id=seed_paper_id)
    else:
        result = workflow.run(user_query=args.query)
    # 归档必须保留本次 baseline 的模式，便于后续把图谱、论文向量、分块向量结果做公平对比。
    result.setdefault("retrieval_mode", retrieval_mode)
    archive_path = archive_chat_idea_result(
        result,
        settings,
        user_query=args.query,
        retrieval_mode=retrieval_mode,
        output_dir=getattr(args, "archive_output_dir", None),
    )
    idea_output = result.get("idea_output")
    if hasattr(idea_output, "model_dump"):
        idea_output = idea_output.model_dump()
    idea_output = idea_output or {}
    canonical_idea = result.get("canonical_idea")
    if hasattr(canonical_idea, "model_dump"):
        canonical_idea = canonical_idea.model_dump()
    canonical_idea = canonical_idea or {}
    query_keywords = result.get("query_keywords") or []
    def _print_evaluation(eval_data, title):
        print(f"\n--- {title} ---")
        print(f"Overall Score: {eval_data.get('overall_score', 'N/A')}/5")
        print(f"Judgment:\n{_normalize_cli_text(eval_data.get('summary', 'N/A'))}")
        for label, key in [
            ("Novelty", "novelty"),
            ("Clarity", "clarity"),
            ("Feasibility", "feasibility"),
            ("Relevance", "relevance"),
            ("Expected Effectiveness", "expected_effectiveness"),
            ("Evidence Grounding", "evidence_grounding"),
        ]:
            criterion = eval_data.get(key) or {}
            print(f"{label}: {criterion.get('score', 'N/A')}/5")
            if criterion.get("comment"):
                print(f"  Comment: {_normalize_cli_text(criterion['comment'])}")
            if criterion.get("revision_suggestion"):
                print(f"  Revision: {_normalize_cli_text(criterion['revision_suggestion'])}")
        for item in _normalize_cli_text_list(eval_data.get("strengths", [])):
            print(f"Strength: {item}")
        for issue in _normalize_cli_text_list(eval_data.get("major_issues", [])):
            print(f"Issue: {issue}")
        ri = eval_data.get("revision_instruction")
        if ri:
            print(f"Revision Instruction:\n{_normalize_cli_text(ri)}")

    def _print_idea_full(idea_data, eval_data, title_label):
        if query_keywords:
            print(f"Query Keywords: {', '.join(query_keywords)}")
            print(f"Retrieval Query: {result.get('retrieval_query', 'N/A')}")
        title = idea_data.get("title") or idea_data.get("Title") or canonical_idea.get("title", "N/A")
        summary = idea_data.get("summary")
        motivation = idea_data.get("problem") or idea_data.get("Motivation") or canonical_idea.get("motivation")
        method = idea_data.get("method")
        experiment = idea_data.get("experiment") or idea_data.get("Experiment") or canonical_idea.get("experiment")
        print(f"Title: {_normalize_cli_text(title)}")
        if summary:
            print(f"Summary:\n{_normalize_cli_text(summary)}")
        if motivation:
            motivation_label = "Motivation" if "Motivation" in idea_data and "problem" not in idea_data else "Problem"
            print(f"{motivation_label}:\n{_normalize_cli_text(motivation)}")
        if method:
            print(f"Method:\n{_normalize_cli_text(method)}")
        if experiment:
            print(f"Experiment:\n{_normalize_cli_text(experiment)}")
        if idea_data.get("idea_lineage"):
            print(f"Idea Lineage:\n{_normalize_cli_text(idea_data['idea_lineage'])}")
        if eval_data:
            _print_evaluation(eval_data, f"{title_label} Evaluation")

    initial_idea = result.get("initial_idea_output")
    if hasattr(initial_idea, "model_dump"):
        initial_idea = initial_idea.model_dump()
    final_eval = result.get("idea_evaluation")
    if hasattr(final_eval, "model_dump"):
        final_eval = final_eval.model_dump()
    comparison = result.get("idea_comparison")
    if hasattr(comparison, "model_dump"):
        comparison = comparison.model_dump()

    # 初版结果（完整格式）
    print("\n--- Initial Idea Generation Result ---")
    if initial_idea:
        _print_idea_full(initial_idea, None, "Initial Idea")

    # 终版结果（完整格式）
    print("\n--- Final Idea Generation Result ---")
    _print_idea_full(idea_output, final_eval, "Final Idea")

    # 初版 vs 终版排序比较
    if comparison:
        label_map = {
            "idea_a": "初始版（Idea A）",
            "idea_b": "改进版（Idea B）",
            "tie": "平手",
        }
        print("\n--- Initial vs Improved: Ranking Comparison ---")
        print(f"Overall Winner: {label_map.get(comparison.get('overall_first', '?'), '?')}")
        print(f"Overall Explanation:\n  {_normalize_cli_text(comparison.get('overall_explanation', ''))}")
        for label, key in [
            ("Novelty", "novelty"),
            ("Clarity", "clarity"),
            ("Feasibility", "feasibility"),
            ("Relevance", "relevance"),
            ("Expected Effectiveness", "expected_effectiveness"),
            ("Evidence Grounding", "evidence_grounding"),
        ]:
            criterion = comparison.get(key) or {}
            winner = label_map.get(criterion.get("first", "?"), "?")
            explanation = criterion.get("explanation", "")
            print(f"\n{label}: {winner}")
            if explanation:
                print(f"  {_normalize_cli_text(explanation)}")
    elif initial_idea:
        # 无比较结果时保持原有简单对比
        print("\n--- Initial Idea vs Final Idea Comparison ---")
        print(f"Title:\n  Before: {_normalize_cli_text(initial_idea.get('title', 'N/A'))}\n  After : {_normalize_cli_text(idea_output.get('title', 'N/A'))}")
        if final_eval:
            print(f"Final Score: {final_eval.get('overall_score', 'N/A')}/5")
    print(
        "Core Paper: "
        f"{idea_output.get('core_paper_title', 'N/A')} "
        f"({idea_output.get('core_paper_id', 'N/A')})"
    )
    print(f"Archived Result: {archive_path}")
    print("------------------------------\n")
    title_for_log = idea_output.get("title") or idea_output.get("Title") or canonical_idea.get("title", "N/A")
    logger.info("chat_idea:result title=%s score=%s", title_for_log[:60], final_eval.get("overall_score") if final_eval else "N/A")
    logger.info("chat_idea:archive path=%s", archive_path)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "build-graph":
            run_build_graph(args)
        elif args.command == "chat-idea":
            run_chat_idea(args)
        elif args.command == "build-openalex-seed-graph":
            run_build_openalex_seed_graph(args)
        elif args.command == "benchmark-accepted-ideas":
            run_benchmark_accepted_ideas(args)
        elif args.command == "prepare-ai-idea-bench-pilot":
            run_prepare_ai_idea_bench_pilot(args)
        elif args.command == "build-local-paper-graph":
            run_build_local_paper_graph(args)
        elif args.command == "preview-entity-extraction":
            run_preview_entity_extraction(args)
    except Exception:
        # 顶层兜底只负责把完整 traceback 写入日志，异常仍交给 CLI 正常失败退出。
        logger.exception("command:failed command=%s", getattr(args, "command", "unknown"))
        raise


if __name__ == "__main__":
    main()
