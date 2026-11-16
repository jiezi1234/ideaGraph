from __future__ import annotations

import json
import logging

from pydantic import ValidationError

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised in tests when dependency is unavailable
    raise RuntimeError("langgraph is required")

from src.agent.generation_profiles import GenerationProfile
from src.agent.generation_profiles import resolve_generation_profile
from src.agent.graph_context import empty_graph_context as _empty_graph_context
from src.agent.graph_context import has_graph_context as _has_graph_context
from src.agent.graph_context import merge_graph_contexts as _merge_graph_contexts
from src.agent.message_utils import extract_json_from_content as _extract_json_from_content
from src.agent.nodes.idea_comparison import idea_comparison
from src.agent.nodes.idea_evaluation import idea_evaluation
from src.agent.nodes.paper_selection import paper_selection
from src.agent.nodes.query_understanding import DEFAULT_QUERY_UNDERSTANDING_SYSTEM_PROMPT, query_understanding
from src.agent.state import AgentState, AgentStateUpdate
from src.observability.langfuse import langfuse_observation
from src.observability.langfuse import serialize_for_langfuse
from src.prompts import CHAT_IDEA_AGENT_PROMPT
from src.prompts import COMPARE_RESEARCH_IDEA_PROMPT
from src.prompts import IMPROVE_IDEA_AGENT_PROMPT
from src.retrieval.base import PaperRetriever
from src.retrieval.selector import TopScorePaperSelector
from src.schemas.agent import CandidatePaper
from src.tools.chat_idea import create_chat_idea_tools
from src.config.settings import Settings
from src.tools.graph.node_neighborhood import TOOL_NAME as GRAPH_NEIGHBORHOOD_TOOL_NAME

logger = logging.getLogger(__name__)

settings = Settings()

DEFAULT_CHAT_IDEA_AGENT_SYSTEM_PROMPT = CHAT_IDEA_AGENT_PROMPT
DEFAULT_IMPROVE_IDEA_AGENT_SYSTEM_PROMPT = IMPROVE_IDEA_AGENT_PROMPT

_MAX_GRAPH_REACT_ROUNDS = 10
_INITIAL_RELATED_PAPER_LIMIT = 3
_INITIAL_RELATED_ENTITY_LIMIT = 5


def build_idea_chat_graph(
    retriever: PaperRetriever,
    selector: TopScorePaperSelector,
    model,
    retrieval_top_k: int,
    graph_context_provider=None,
    initial_entity_retriever=None,
    context_chunk_retriever=None,
    query_understanding_prompt_template: str | None = None,
    agent_prompt_template: str | None = None,
    evaluation_prompt_template: str | None = None,
    evaluation_model=None,
    generation_profile: str = "ideagraph",
    retrieval_mode: str = "graph",
    seed_paper_provider=None,
    langfuse_client=None,
):
    generation_profile_config = resolve_generation_profile(generation_profile)
    _validate_retrieval_mode(retrieval_mode)
    internal_evaluation_prompt_template = (
        evaluation_prompt_template if generation_profile_config.internal_evaluation_enabled else None
    )
    active_graph_context_provider = graph_context_provider if retrieval_mode == "graph" else None
    active_initial_entity_retriever = initial_entity_retriever if retrieval_mode == "graph" else None
    active_evaluation_model = evaluation_model or model
    graph = StateGraph(AgentState)

    def paper_retrieval(state: AgentState) -> AgentStateUpdate:
        input_payload = {
            "retrieval_query": state["retrieval_query"],
            "retrieval_top_k": retrieval_top_k,
        }
        with langfuse_observation(
            langfuse_client,
            name="paper_retrieval",
            as_type="retriever",
            input=input_payload,
            metadata={"stage": "retrieval", "retrieval_top_k": retrieval_top_k},
        ) as span:
            candidate_papers = retriever.retrieve(state["retrieval_query"], retrieval_top_k)
            if span is not None:
                span.update(
                    output={"candidate_papers": serialize_for_langfuse(candidate_papers[:5])},
                    metadata={
                        "stage": "retrieval",
                        "retrieval_top_k": retrieval_top_k,
                        "candidate_count": len(candidate_papers),
                    },
                )
        if candidate_papers:
            return {"candidate_papers": candidate_papers}
        return {
            "candidate_papers": [],
            "errors": [f"No candidate papers retrieved for query: {state['retrieval_query']}"],
        }

    def seed_paper_selection(state: AgentState) -> AgentStateUpdate:
        seed_paper_id = (state.get("seed_paper_id") or "").strip()
        input_payload = {"seed_paper_id": seed_paper_id}
        with langfuse_observation(
            langfuse_client,
            name="seed_paper_selection",
            as_type="retriever",
            input=input_payload,
            metadata={"stage": "seed_paper_selection"},
        ) as span:
            if not seed_paper_id:
                update = {"errors": ["Seed paper id is empty"]}
            elif seed_paper_provider is None:
                update = {"errors": [f"Seed paper lookup is unavailable: {seed_paper_id}"]}
            else:
                paper = seed_paper_provider.get_paper_by_id(seed_paper_id)
                if paper is None:
                    update = {
                        "candidate_papers": [],
                        "errors": [f"Seed paper not found in graph: {seed_paper_id}"],
                    }
                else:
                    candidate = CandidatePaper(
                        paper_id=paper.document_id,
                        title=paper.title,
                        abstract=paper.abstract,
                        cited_by_count=paper.cited_by_count,
                        score=1.0,
                    )
                    update = {
                        "retrieval_query": state["user_query"].strip(),
                        "query_keywords": [],
                        "candidate_papers": [candidate],
                        "selected_paper": candidate,
                    }
            if span is not None:
                span.update(
                    output=serialize_for_langfuse(update),
                    metadata={
                        "stage": "seed_paper_selection",
                        "seed_paper_found": bool(update.get("selected_paper")),
                    },
                )
            return update

    def paper_selection_node(state: AgentState) -> AgentStateUpdate:
        if state.get("errors"):
            return {"errors": state["errors"]}
        input_payload = {
            "user_query": state["retrieval_query"],
            "candidate_papers": serialize_for_langfuse(state["candidate_papers"]),
        }
        with langfuse_observation(
            langfuse_client,
            name="paper_selection",
            as_type="chain",
            input=input_payload,
            metadata={"stage": "selection", "candidate_count": len(state["candidate_papers"])},
        ) as span:
            update = paper_selection(
                {
                    "user_query": state["retrieval_query"],
                    "candidate_papers": state["candidate_papers"],
                },
                selector=selector,
            )
            if span is not None:
                span.update(
                    output={"selected_paper": serialize_for_langfuse(update.get("selected_paper"))},
                    metadata={
                        "stage": "selection",
                        "selected_paper_id": getattr(update.get("selected_paper"), "paper_id", None),
                    },
                )
            return update

    def related_entity_retrieval_node(state: AgentState) -> AgentStateUpdate:
        """初始阶段只给 agent 实体 ID 种子；具体描述和关系由后续图谱工具查询。"""
        if state.get("errors") or active_initial_entity_retriever is None:
            return {}
        query = state.get("retrieval_query") or state["user_query"]
        input_payload = {"retrieval_query": query, "top_k": _INITIAL_RELATED_ENTITY_LIMIT}
        with langfuse_observation(
            langfuse_client,
            name="related_entity_retrieval",
            as_type="retriever",
            input=input_payload,
            metadata={"stage": "related_entity_retrieval", "top_k": _INITIAL_RELATED_ENTITY_LIMIT},
        ) as span:
            related_entities = _serialize_related_entity_ids(
                active_initial_entity_retriever.retrieve(query, _INITIAL_RELATED_ENTITY_LIMIT)
            )
            if span is not None:
                span.update(
                    output={"related_entities": serialize_for_langfuse(related_entities)},
                    metadata={
                        "stage": "related_entity_retrieval",
                        "top_k": _INITIAL_RELATED_ENTITY_LIMIT,
                        "entity_count": len(related_entities),
                    },
                )
            return {"related_entities": related_entities}

    def idea_evaluation_node(state: AgentState) -> AgentStateUpdate:
        if state.get("errors") or internal_evaluation_prompt_template is None:
            return {}
        update = idea_evaluation(
            state,
            model=active_evaluation_model,
            prompt_template=internal_evaluation_prompt_template,
            graph_context_provider=active_graph_context_provider,
        )
        update["iteration"] = state.get("iteration", 0) + 1
        return update

    def idea_comparison_node(state: AgentState) -> AgentStateUpdate:
        if state.get("errors") or not state.get("initial_idea_output"):
            return {}
        return idea_comparison(
            state,
            model=model,
            prompt_template=COMPARE_RESEARCH_IDEA_PROMPT,
        )

    def query_understanding_node(state: AgentState) -> AgentStateUpdate:
        return query_understanding(
            {"user_query": state["user_query"]},
            model=model,
            prompt_template=query_understanding_prompt_template or DEFAULT_QUERY_UNDERSTANDING_SYSTEM_PROMPT,
        )

    def start_route(state: AgentState) -> str:
        if (state.get("seed_paper_id") or "").strip():
            return "seed_paper_selection"
        return "query_understanding"

    def agent_generation_node(state: AgentState) -> AgentStateUpdate:
        if state.get("errors"):
            return {}
        iteration = state.get("iteration", 0)
        if agent_prompt_template is not None:
            prompt_template = agent_prompt_template
        elif iteration > 0 and generation_profile_config.name == "ideagraph":
            prompt_template = DEFAULT_IMPROVE_IDEA_AGENT_SYSTEM_PROMPT
        else:
            prompt_template = generation_profile_config.prompt_template
        input_payload = {
            "user_query": state["user_query"],
            "retrieval_query": state.get("retrieval_query"),
            "query_keywords": state.get("query_keywords", []),
            "generation_profile": generation_profile_config.name,
            "retrieval_mode": retrieval_mode,
            "related_papers": serialize_for_langfuse(state.get("candidate_papers", [])[:_INITIAL_RELATED_PAPER_LIMIT]),
            "related_entities": serialize_for_langfuse(state.get("related_entities", [])),
            "retrieved_chunks": serialize_for_langfuse(state.get("retrieved_chunks", [])),
            "graph_context": state.get("graph_context"),
            "idea_evaluation": serialize_for_langfuse(state.get("idea_evaluation")) if iteration > 0 else None,
        }
        with langfuse_observation(
            langfuse_client,
            name="agent_generation",
            as_type="agent",
            input=input_payload,
            metadata={
                "stage": "agent_generation",
                "iteration": iteration,
                "has_graph_context": bool(state.get("graph_context")),
                "graph_context_entity_count": len((state.get("graph_context") or {}).get("entities", [])),
                "graph_context_relationship_count": len((state.get("graph_context") or {}).get("relationships", [])),
            },
        ) as span:
            generation_state = state
            chunk_update = {}
            if retrieval_mode == "chunk_vector" and state.get("iteration", 0) == 0:
                retrieved_chunks = (
                    context_chunk_retriever.retrieve(state["retrieval_query"], retrieval_top_k)
                    if context_chunk_retriever is not None
                    else []
                )
                chunk_update = {"retrieved_chunks": retrieved_chunks}
                generation_state = {**state, **chunk_update}
            if active_graph_context_provider is None:
                update = _generate_idea_without_tools(
                    generation_state,
                    model,
                    prompt_template=prompt_template,
                    generation_profile=generation_profile_config,
                )
            else:
                update = _run_idea_agent_loop(
                    generation_state,
                    model,
                    active_graph_context_provider,
                    prompt_template=prompt_template,
                    generation_profile=generation_profile_config,
                    langfuse_client=langfuse_client,
                )
            update.update(chunk_update)
            # 保存上一轮 idea_output，供后续对比排序
            if iteration > 0 and state.get("idea_output") is not None:
                update["initial_idea_output"] = state["idea_output"]
            if span is not None:
                span.update(
                    output={
                        "idea_output": serialize_for_langfuse(update.get("idea_output")),
                        "graph_context": serialize_for_langfuse(update.get("graph_context")),
                        "tool_trace": serialize_for_langfuse(update.get("tool_trace", [])),
                    },
                    metadata={
                        "stage": "agent_generation",
                        "has_idea_thinking": bool(update.get("idea_thinking")),
                        "tool_round_count": len(update.get("tool_trace", [])),
                    },
                )
            return update

    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("seed_paper_selection", seed_paper_selection)
    graph.add_node("paper_retrieval", paper_retrieval)
    graph.add_node("paper_selection", paper_selection_node)
    graph.add_node("related_entity_retrieval", related_entity_retrieval_node)
    graph.add_node("agent_generation", agent_generation_node)
    graph.add_node("idea_evaluation", idea_evaluation_node)
    graph.add_node("idea_comparison", idea_comparison_node)

    graph.add_conditional_edges(
        START,
        start_route,
        {
            "query_understanding": "query_understanding",
            "seed_paper_selection": "seed_paper_selection",
        },
    )
    graph.add_edge("query_understanding", "paper_retrieval")
    graph.add_edge("paper_retrieval", "paper_selection")
    graph.add_edge("paper_selection", "related_entity_retrieval")
    graph.add_edge("seed_paper_selection", "related_entity_retrieval")
    graph.add_edge("related_entity_retrieval", "agent_generation")
    graph.add_edge("agent_generation", "idea_evaluation")
    graph.add_conditional_edges(
        "idea_evaluation",
        lambda state: "agent_generation"
        if not state.get("errors") and internal_evaluation_prompt_template is not None and state.get("iteration", 0) < 2
        else "idea_comparison"
        if not state.get("errors") and state.get("initial_idea_output") is not None
        else END,
        {"agent_generation": "agent_generation", "idea_comparison": "idea_comparison", END: END},
    )
    graph.add_edge("idea_comparison", END)

    return graph.compile()


def _run_idea_agent_loop(
    state: AgentState,
    model,
    graph_context_provider,
    prompt_template: str,
    generation_profile: GenerationProfile,
    langfuse_client=None,
) -> AgentStateUpdate:
    tools = create_chat_idea_tools(graph_context_provider)
    tool_by_name = {tool.name: tool for tool in tools}
    openai_tools = [tool.openai_schema() for tool in tools]

    messages = _build_idea_agent_messages(state, prompt_template)
    round_contexts = []
    tool_trace = []
    idea_thinking = ""
    direct_final_update = None

    round_index = 0
    retry_count = 0
    MAX_RETRIES_PER_ROUND = 3
    previous_allowed_nodes = []
    all_allowed_nodes = []
    # 写代码时要加必要的中文注释：记录已被实际成功查询的节点 ID，用于防环校验 (Anti-loop)
    queried_nodes = []
    selected_paper = state.get("selected_paper")
    selected_paper_title = getattr(selected_paper, "title", "") if selected_paper else ""

    while round_index < _MAX_GRAPH_REACT_ROUNDS:
        # 写代码时要加必要的中文注释：前三步（Round 0, 1, 2）查询 n 跳（配置项），后续查询每次执行 1 跳
        current_depth = settings.graph_multihop_depth if round_index < 3 else 1

        with langfuse_observation(
            langfuse_client,
            name=f"agent_generation_round_{round_index}_result",
            as_type="generation",
            input={
                "messages": serialize_for_langfuse(messages[-3:]),
                "tool_count": len(openai_tools),
            },
            metadata={"stage": "agent_generation_round_result", "round": round_index},
        ) as round_span:
            response = model.chat_with_tools(
                messages=messages,
                tools=openai_tools,
                name=f"agent_generation_round_{round_index}",
                metadata={"stage": "agent_generation_round", "round": round_index},
            )
            
            tool_calls = _normalize_openai_tool_calls(response.tool_calls)
            
            # 写代码时要加必要的中文注释：防环校验 (Anti-loop)。如果查询节点已被访问过，则驳回重试
            if tool_calls:
                queried_node = tool_calls[0]["arguments"]["node_id"]
                if any(q.strip().lower() == queried_node.strip().lower() for q in queried_nodes):
                    if retry_count < MAX_RETRIES_PER_ROUND:
                        retry_count += 1
                        logger.warning("Round %s: Queried node '%s' already queried in previous rounds. Retrying... (attempt %s/%s)", round_index, queried_node, retry_count, MAX_RETRIES_PER_ROUND)
                        messages.append(_assistant_message_from_response(response))
                        tool_results = _execute_tool_calls(tool_by_name, response.tool_calls, depth=current_depth)
                        for tool_call, tool_result in zip(response.tool_calls, tool_results, strict=True):
                            messages.append(_tool_result_message(tool_call, tool_result))
                        
                        allowed_list = previous_allowed_nodes if round_index < 3 else all_allowed_nodes
                        allowed_desc = "上一步返回的邻居" if round_index < 3 else "已探索的图谱连通"
                        messages.append({
                            "role": "user",
                            "content": f"你查询的节点 '{queried_node}' 已经在前几轮查询过。为了避免探索陷入环路并拓宽信息面，请从{allowed_desc}节点中选择一个未查询过的节点重新查询。可选的节点有：{', '.join(allowed_list[:10])}。"
                        })
                        continue

            # 校验逻辑 1：第一轮必须执行工具
            if round_index == 0 and not tool_calls:
                if retry_count < MAX_RETRIES_PER_ROUND:
                    retry_count += 1
                    logger.warning("Round 0: Agent did not call tools. Retrying... (attempt %s/%s)", retry_count, MAX_RETRIES_PER_ROUND)
                    messages.append(_assistant_message_from_response(response))
                    messages.append({
                        "role": "user",
                        "content": f"为了启动学术图谱深度探索，第一步你必须调用 get_node_neighborhood 工具。请使用种子论文标题：'{selected_paper_title}' 或是相关的核心学术概念作为 node_id 进行查询。"
                    })
                    continue
                else:
                    logger.warning("Round 0: Max retries reached without tool calls. Proceeding.")
            
            # 校验逻辑 2：第二轮必须执行工具，且必须是上一步返回的邻居
            # 校验逻辑 2：第二轮必须执行工具，且必须是上一步返回的邻居，且数量必须满足限制以保证探索的分支宽度
            elif round_index == 1:
                # 写代码时要加必要的中文注释：计算本轮所需的并行查询宽度，若上一跳生成的邻居候选数少于配置宽度，则向下取齐，避免强制要求多于候选总数的无效查询。
                required_width = min(settings.graph_react_width, len(previous_allowed_nodes)) if previous_allowed_nodes else settings.graph_react_width
                if not tool_calls:
                    if retry_count < MAX_RETRIES_PER_ROUND:
                        retry_count += 1
                        logger.warning("Round 1: Agent did not call tools. Retrying... (attempt %s/%s)", retry_count, MAX_RETRIES_PER_ROUND)
                        messages.append(_assistant_message_from_response(response))
                        messages.append({
                            "role": "user",
                            "content": f"为了保证图谱检索的广度与深度，在第 2 步你必须沿着刚才的方向继续进行并行查询，且选择正好 {required_width} 个不同的方向（节点）。请从上一步返回的邻域中挑选 {required_width} 个节点作为 node_id 调用 get_node_neighborhood 工具。可选的邻域节点有：{', '.join(previous_allowed_nodes[:10])}。"
                        })
                        continue
                else:
                    # 写代码时要加必要的中文注释：遍历本轮所有的工具调用 node_id，确保模型如果一次性并行查询多个节点，所有的查询节点都必须在合法邻域中，避免局部越界遗漏校验。
                    invalid_nodes = [
                        tc["arguments"].get("node_id")
                        for tc in tool_calls
                        if previous_allowed_nodes and not _is_node_in_list(tc["arguments"].get("node_id"), previous_allowed_nodes)
                    ]
                    if len(tool_calls) != required_width or invalid_nodes:
                        if retry_count < MAX_RETRIES_PER_ROUND:
                            retry_count += 1
                            if len(tool_calls) != required_width:
                                logger.warning("Round 1: Agent did not call exactly %d tools (got %d). Retrying...", required_width, len(tool_calls))
                                error_msg = f"为了保证探索的广度，你必须同时选择正好 {required_width} 个不同的方向（节点）进行并行查询（当前你查询了 {len(tool_calls)} 个）。请从上一步返回的邻域中挑选 {required_width} 个节点作为 node_id 调用 get_node_neighborhood 工具。可选的邻域节点有：{', '.join(previous_allowed_nodes[:10])}。"
                            else:
                                if len(invalid_nodes) == 1:
                                    node_str = f"node '{invalid_nodes[0]}'"
                                    node_msg = f"节点 '{invalid_nodes[0]}'"
                                else:
                                    node_str = f"nodes {invalid_nodes}"
                                    node_msg = f"节点 {invalid_nodes}"
                                logger.warning("Round 1: Queried %s not in previous neighborhood. Retrying...", node_str)
                                error_msg = f"你查询的{node_msg}不在上一跳的邻域中。请重新选择刚好 {required_width} 个可选的邻域节点进行 get_node_neighborhood 查询。可选的邻域节点有：{', '.join(previous_allowed_nodes[:10])}。"
                            
                            messages.append(_assistant_message_from_response(response))
                            tool_results = _execute_tool_calls(tool_by_name, response.tool_calls, depth=current_depth)
                            for tool_call, tool_result in zip(response.tool_calls, tool_results, strict=True):
                                messages.append(_tool_result_message(tool_call, tool_result))
                            messages.append({
                                "role": "user",
                                "content": error_msg
                            })
                            continue
            
            # 校验逻辑 3：第三轮必须执行工具，且必须是上一步返回的邻居，且数量必须满足限制以保证探索的分支宽度
            elif round_index == 2:
                # 写代码时要加必要的中文注释：计算本轮所需的并行查询宽度，若上一跳生成的邻居候选数少于配置宽度，则向下取齐，避免强制要求多于候选总数的无效查询。
                required_width = min(settings.graph_react_width, len(previous_allowed_nodes)) if previous_allowed_nodes else settings.graph_react_width
                if not tool_calls:
                    if retry_count < MAX_RETRIES_PER_ROUND:
                        retry_count += 1
                        logger.warning("Round 2: Agent did not call tools. Retrying... (attempt %s/%s)", retry_count, MAX_RETRIES_PER_ROUND)
                        messages.append(_assistant_message_from_response(response))
                        messages.append({
                            "role": "user",
                            "content": f"你必须进行第三跳的查询以完成深度证据链探索。请从上一步返回的邻域中挑选 {required_width} 个节点作为 node_id 调用 get_node_neighborhood 工具。可选的邻域节点有：{', '.join(previous_allowed_nodes[:10])}。"
                        })
                        continue
                else:
                    # 写代码时要加必要的中文注释：遍历本轮所有的工具调用 node_id，确保模型如果一次性并行查询多个节点，所有的查询节点都必须在合法邻域中，避免局部越界遗漏校验。
                    invalid_nodes = [
                        tc["arguments"].get("node_id")
                        for tc in tool_calls
                        if previous_allowed_nodes and not _is_node_in_list(tc["arguments"].get("node_id"), previous_allowed_nodes)
                    ]
                    if len(tool_calls) != required_width or invalid_nodes:
                        if retry_count < MAX_RETRIES_PER_ROUND:
                            retry_count += 1
                            if len(tool_calls) != required_width:
                                logger.warning("Round 2: Agent did not call exactly %d tools (got %d). Retrying...", required_width, len(tool_calls))
                                error_msg = f"为了保证探索的广度，你必须同时选择正好 {required_width} 个不同的方向（节点）进行并行查询（当前你查询了 {len(tool_calls)} 个）。请从上一步返回的邻域中挑选 {required_width} 个节点作为 node_id 调用 get_node_neighborhood 工具。可选的邻域节点有：{', '.join(previous_allowed_nodes[:10])}。"
                            else:
                                if len(invalid_nodes) == 1:
                                    node_str = f"node '{invalid_nodes[0]}'"
                                    node_msg = f"节点 '{invalid_nodes[0]}'"
                                else:
                                    node_str = f"nodes {invalid_nodes}"
                                    node_msg = f"节点 {invalid_nodes}"
                                logger.warning("Round 2: Queried %s not in previous neighborhood. Retrying...", node_str)
                                error_msg = f"你查询的{node_msg}不在上一步的邻域中。请重新选择刚好 {required_width} 个可选的邻域节点进行 get_node_neighborhood 查询。可选的邻域节点有：{', '.join(previous_allowed_nodes[:10])}。"
                            
                            messages.append(_assistant_message_from_response(response))
                            tool_results = _execute_tool_calls(tool_by_name, response.tool_calls, depth=current_depth)
                            for tool_call, tool_result in zip(response.tool_calls, tool_results, strict=True):
                                messages.append(_tool_result_message(tool_call, tool_result))
                            messages.append({
                                "role": "user",
                                "content": error_msg
                            })
                            continue

            elif round_index >= 3 and tool_calls:
                # 写代码时要加必要的中文注释：遍历本轮所有的工具调用 node_id，确保模型如果一次性并行查询多个节点，所有的查询节点都必须在历史已探索的图谱连通区域内。
                invalid_nodes = [
                    tc["arguments"].get("node_id")
                    for tc in tool_calls
                    if all_allowed_nodes and not _is_node_in_list(tc["arguments"].get("node_id"), all_allowed_nodes)
                ]
                if invalid_nodes:
                    if retry_count < MAX_RETRIES_PER_ROUND:
                        retry_count += 1
                        if len(invalid_nodes) == 1:
                            node_str = f"node '{invalid_nodes[0]}'"
                            node_msg = f"节点 '{invalid_nodes[0]}'"
                        else:
                            node_str = f"nodes {invalid_nodes}"
                            node_msg = f"节点 {invalid_nodes}"
                        logger.warning("Round %s: Queried %s not in history neighborhood. Retrying...", round_index, node_str)
                        messages.append(_assistant_message_from_response(response))
                        tool_results = _execute_tool_calls(tool_by_name, response.tool_calls, depth=current_depth)
                        for tool_call, tool_result in zip(response.tool_calls, tool_results, strict=True):
                            messages.append(_tool_result_message(tool_call, tool_result))
                        messages.append({
                            "role": "user",
                            "content": f"你查询的{node_msg}不在已探索的图谱连通区域内。虽然 3 步强制探索已完成，但后续查询仍需保证在图谱范围内连通。你可以通过回溯（Backtracking）选择之前访问过的任何节点及其邻域。可选的关联节点有：{', '.join(all_allowed_nodes[:10])}。"
                        })
                        continue

            messages.append(_assistant_message_from_response(response))
            if response.reasoning:
                idea_thinking = response.reasoning

            if round_span is not None:
                update_kwargs = {
                    "output": {
                        "content": response.content or "",
                        "reasoning": response.reasoning or "",
                        "finish_reason": response.finish_reason or "",
                        "tool_calls": serialize_for_langfuse(tool_calls),
                    },
                    "metadata": {
                        "stage": "agent_generation_round_result",
                        "round": round_index,
                        "has_content": bool((response.content or "").strip()),
                        "has_reasoning": bool(response.reasoning),
                        "finish_reason": response.finish_reason or "",
                        "tool_call_count": len(tool_calls),
                    },
                }
                round_span.update(**update_kwargs)

        if not tool_calls:
            direct_final_update = _parse_direct_idea_agent_response(
                response,
                round_index=round_index,
                generation_profile=generation_profile,
            )
            break

        _log_planned_graph_tool_calls(tool_calls, round_index=round_index)
        with langfuse_observation(
            langfuse_client,
            name=f"graph_tool_execution_round_{round_index}",
            as_type="tool",
            input={"tool_calls": serialize_for_langfuse(tool_calls)},
            metadata={"stage": "graph_tool_execution", "round": round_index},
        ) as span:
            tool_results = _execute_tool_calls(tool_by_name, response.tool_calls, depth=current_depth)
            for tool_call, tool_result in zip(response.tool_calls, tool_results, strict=True):
                messages.append(_tool_result_message(tool_call, tool_result))
            if span is not None:
                span.update(
                    output=serialize_for_langfuse(tool_results),
                    metadata={
                        "stage": "graph_tool_execution",
                        "round": round_index,
                        "tool_message_count": len(tool_results),
                    },
                )

        round_context = _merge_graph_contexts(tool_results)
        _log_done_graph_tool_calls(tool_calls, round_context)
        round_contexts.append(round_context)
        tool_trace.append(
            {
                "round": round_index,
                "tool_calls": tool_calls,
                "observation": round_context,
            }
        )

        # 写代码时要加必要的中文注释：记录本次成功且未报错的查询节点到防环已访问列表中
        for tc, tr in zip(tool_calls, tool_results, strict=True):
            if tr.get("status") == "error":
                continue
            q_id = tc["arguments"].get("node_id")
            if q_id and q_id.strip().lower() not in queried_nodes:
                queried_nodes.append(q_id.strip().lower())

        # 记录本次成功查询产生的邻居节点，并合并入全局允许探索列表
        queried_in_round = [tc["arguments"]["node_id"] for tc in tool_calls]
        new_neighbors = []
        for entity in round_context.get("entities") or []:
            ent_id = entity.get("id")
            if ent_id and not any(q.strip().lower() == ent_id.strip().lower() for q in queried_in_round):
                new_neighbors.append(ent_id)

        previous_allowed_nodes = new_neighbors

        for n in new_neighbors:
            if not any(n.strip().lower() == existing.strip().lower() for existing in all_allowed_nodes):
                all_allowed_nodes.append(n)

        round_index += 1
        retry_count = 0
    else:
        logger.info("graph_tool_react:max_rounds_reached rounds=%s", _MAX_GRAPH_REACT_ROUNDS)

    current_graph_context = _merge_graph_contexts(round_contexts)
    # 改进轮会重新开模型对话，但上一轮图谱证据仍是改进依据，不能被空查询覆盖。
    graph_context = _merge_previous_and_current_graph_context(state, current_graph_context)
    combined_tool_trace = _merge_previous_and_current_tool_trace(state, tool_trace)
    final_update = direct_final_update or _generate_final_idea_with_structured_output(
        state=state,
        model=model,
        messages=messages,
        graph_context=graph_context,
        generation_profile=generation_profile,
    )
    update = {
        "graph_context": graph_context,
        "tool_trace": combined_tool_trace,
    }
    update.update(_generation_output_update(final_update["idea_output"], generation_profile))
    if final_update.get("idea_thinking"):
        update["idea_thinking"] = final_update["idea_thinking"]
    elif idea_thinking:
        update["idea_thinking"] = idea_thinking
    return update


def _build_idea_agent_messages(state: AgentState, prompt_template: str) -> list[dict]:
    query_keywords = state.get("query_keywords", [])
    keywords_line = f"Query keywords: {', '.join(query_keywords)}\n" if query_keywords else ""
    content = (
        "Current phase: idea_generation_with_tools\n"
        "Use the provided native tool API when graph context can improve the final idea. "
        "If no graph lookup is useful, produce the final idea JSON directly.\n"
        f"User query: {state['user_query']}\n"
        f"Retrieval query: {state.get('retrieval_query', '')}\n"
        f"{keywords_line}"
        f"{_format_related_papers_block(state, limit=_INITIAL_RELATED_PAPER_LIMIT)}\n"
        f"{_format_related_entity_ids_block(state.get('related_entities') or [])}\n"
    )
    content += _improvement_prompt_context(state)
    content += _previous_graph_prompt_context(state)

    messages = [{"role": "system", "content": prompt_template}]
    messages.append({
        "role": "user",
        "content": content,
    })
    return messages


def _improvement_prompt_context(state: AgentState) -> str:
    if state.get("iteration", 0) <= 0 or not state.get("idea_evaluation"):
        return ""
    return (
        "\nInitial idea:\n"
        f"{json.dumps(serialize_for_langfuse(state.get('idea_output')), ensure_ascii=False, indent=2)}\n"
        "Initial idea evaluation:\n"
        f"{json.dumps(serialize_for_langfuse(state['idea_evaluation']), ensure_ascii=False, indent=2)}"
    )


def _previous_graph_prompt_context(state: AgentState) -> str:
    if state.get("iteration", 0) <= 0:
        return ""
    sections = []
    previous_graph_context = state.get("graph_context") or {}
    if _has_graph_context(previous_graph_context):
        sections.append(
            "Previous graph context from the initial generation:\n"
            f"{_format_graph_context_human_readable(previous_graph_context)}"
        )
    previous_tool_trace = state.get("tool_trace") or []
    if previous_tool_trace:
        sections.append(
            "Previous graph tool trace from the initial generation:\n"
            f"{_format_tool_trace_human_readable(previous_tool_trace)}"
        )
    if not sections:
        return ""
    return "\n" + "\n".join(sections)


def _parse_direct_idea_agent_response(
    response,
    *,
    round_index: int,
    generation_profile: GenerationProfile,
) -> AgentStateUpdate | None:
    content = _extract_json_from_content(response.content or "")
    logger.info(
        "graph_tool_react:final_response round=%s has_content=%s",
        round_index,
        bool(content),
    )
    if not content:
        return None
    try:
        idea_output = generation_profile.output_schema.model_validate_json(content)
    except ValidationError:
        logger.warning(
            "graph_tool_react:direct_final_response_invalid round=%s; falling back to structured generation",
            round_index,
        )
        return None
    logger.info("graph_tool_react:use_direct_final_output round=%s", round_index)
    update = _generation_output_update(idea_output, generation_profile)
    if response.reasoning:
        update["idea_thinking"] = response.reasoning
    return update


def _generate_final_idea_with_structured_output(
    state: AgentState,
    model,
    messages: list[dict],
    graph_context: dict,
    generation_profile: GenerationProfile,
) -> AgentStateUpdate:
    content = (
        "Current phase: final_idea_generation\n"
        f"Generate the final idea now. Return only the structured {generation_profile.output_schema.__name__} fields."
    )
    content += _improvement_prompt_context(state)
    result = model.structured_json(
        messages=[
            *messages,
            {
                "role": "user",
                "content": content,
            },
        ],
        schema=generation_profile.output_schema,
        name="final_idea_generation",
        metadata={"stage": "final_idea_generation", "generation_profile": generation_profile.name},
    )
    parsed = result.parsed
    idea_output = (
        parsed if isinstance(parsed, generation_profile.output_schema) else generation_profile.output_schema(**parsed)
    )
    update = _generation_output_update(idea_output, generation_profile)
    if result.reasoning:
        update["idea_thinking"] = result.reasoning
    return update


def _generate_idea_without_tools(
    state: AgentState,
    model,
    prompt_template: str,
    generation_profile: GenerationProfile,
) -> AgentStateUpdate:
    content = (
        "Current phase: final_idea_generation\n"
        f"User query: {state['user_query']}\n"
        f"Retrieval query: {state.get('retrieval_query', '')}\n"
        f"{_format_related_papers_block(state, limit=_INITIAL_RELATED_PAPER_LIMIT)}\n"
        f"{_format_related_entity_ids_block(state.get('related_entities') or [])}"
    )
    content += _retrieved_chunks_prompt_context(state)
    content += _improvement_prompt_context(state)
    result = model.structured_json(
        messages=[
            {"role": "system", "content": prompt_template},
            {
                "role": "user",
                "content": content,
            },
        ],
        schema=generation_profile.output_schema,
        name="final_idea_generation",
        metadata={"stage": "final_idea_generation", "generation_profile": generation_profile.name},
    )
    parsed = result.parsed
    idea_output = (
        parsed if isinstance(parsed, generation_profile.output_schema) else generation_profile.output_schema(**parsed)
    )
    update = _generation_output_update(idea_output, generation_profile)
    if result.reasoning:
        update["idea_thinking"] = result.reasoning
    return update


def _generation_output_update(idea_output, generation_profile: GenerationProfile) -> AgentStateUpdate:
    return {
        "idea_output": idea_output,
        "canonical_idea": generation_profile.canonicalize(idea_output),
        "generation_profile": generation_profile.name,
        "prompt_variant": generation_profile.prompt_variant,
    }


def _validate_retrieval_mode(retrieval_mode: str) -> None:
    if retrieval_mode not in {"graph", "paper_vector", "chunk_vector"}:
        raise ValueError(
            "retrieval_mode must be one of: graph, paper_vector, chunk_vector"
        )


def _retrieved_chunks_prompt_context(state: AgentState) -> str:
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return ""
    lines = ["\nRetrieved vector context:"]
    for index, chunk in enumerate(chunks, start=1):
        lines.extend(
            [
                f"[{index}] Paper: {chunk.title} ({chunk.paper_id})",
                f"Score: {chunk.score}",
                f"Text source: {chunk.text_source}",
                f"Chunk id: {chunk.chunk_id}",
                "Chunk:",
                chunk.text,
            ]
        )
    return "\n".join(lines)


def _format_candidate_papers_for_tool_planning(state: AgentState, *, limit: int = 5) -> list[dict]:
    candidates = []
    for paper in state.get("candidate_papers", [])[:limit]:
        candidates.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "score": paper.score,
            }
        )
    return candidates


def _format_related_papers_block(state: AgentState, *, limit: int) -> str:
    """格式化 top-N 相关论文块，带摘要，供 prompt 使用。"""
    candidates = _format_candidate_papers_for_tool_planning(state, limit=limit)
    if not candidates:
        return "Related papers (top 3 with abstracts):\n  (none)"
    lines = ["Related papers (top 3 with abstracts):"]
    for index, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"[{index}] {candidate['title']} ({candidate['paper_id']}, score: {candidate['score']})",
                f"Abstract: {candidate['abstract']}",
            ]
        )
    return "\n".join(lines)


def _format_related_entity_ids_block(related_entities: list[dict]) -> str:
    """格式化初始相关实体 ID 列表，供 prompt 使用。"""
    if not related_entities:
        return "Initial related entity ids (top 5):\n  (none)"
    lines = ["Initial related entity ids (top 5):"]
    for entity in related_entities[:_INITIAL_RELATED_ENTITY_LIMIT]:
        entity_id = str(entity.get("id") or "").strip()
        if entity_id:
            lines.append(f"  - {entity_id}")
    if len(lines) == 1:
        lines.append("  (none)")
    return "\n".join(lines)


def _serialize_related_entity_ids(entities: list[dict]) -> list[dict]:
    """去重并截断实体 ID 列表。"""
    serialized = []
    seen = set()
    for entity in entities:
        entity_id = str(entity.get("id") or "").strip()
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        serialized.append({"id": entity_id})
        if len(serialized) >= _INITIAL_RELATED_ENTITY_LIMIT:
            break
    return serialized


def _normalize_openai_tool_calls(tool_calls: list) -> list[dict]:
    """归一化 OpenAI tool_calls，保留原始 arguments，不做工具特定的字段假设。"""
    normalized = []
    for tool_call in tool_calls or []:
        args = dict(getattr(tool_call, "arguments", {}) or {})
        normalized.append(
            {
                "tool_name": getattr(tool_call, "tool_name", ""),
                "tool_call_id": getattr(tool_call, "tool_call_id", ""),
                "arguments": args,
            }
        )
    return normalized


def _assistant_message_from_response(response) -> dict:
    message = {
        "role": "assistant",
        "content": response.content or "",
    }
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for tool_call in response.tool_calls
        ]
    return message


def _tool_result_message(tool_call, tool_result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call.tool_call_id,
        "name": tool_call.tool_name,
        "content": json.dumps(tool_result, ensure_ascii=False, sort_keys=True),
    }


def _execute_tool_calls(tool_by_name: dict, tool_calls: list, depth: int = 1) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        tool = tool_by_name.get(tool_call.tool_name)
        if tool is None:
            error = ValueError(f"Unknown graph tool: {tool_call.tool_name}")
            results.append(_tool_error_result(tool_call, error))
            logger.warning(
                "graph_tool_call:error tool=%s",
                tool_call.tool_name,
                exc_info=(type(error), error, error.__traceback__),
            )
            continue
        try:
            # 写代码时要加必要的中文注释：在底层执行工具调用时，如果是 get_node_neighborhood 且指定了 depth，则动态注入 depth
            arguments = dict(tool_call.arguments)
            if tool_call.tool_name == "get_node_neighborhood":
                arguments["depth"] = depth
            results.append(tool.call(arguments))
        except Exception as error:
            # 工具调用失败也要回传给模型，让下一轮能修正参数或停止调用。
            results.append(_tool_error_result(tool_call, error))
            logger.warning("graph_tool_call:error tool=%s", tool_call.tool_name, exc_info=True)
    return results


def _tool_error_result(tool_call, error: Exception) -> dict:
    return {
        "status": "error",
        "tool_name": tool_call.tool_name,
        "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "recoverable": True,
        "instruction": (
            "The tool call failed. Reconsider the tool arguments using the tool schema, "
            "then retry with corrected arguments or continue without this tool if the context is sufficient."
        ),
        "entities": [],
        "relationships": [],
        "page": 1,
        "page_size": 20,
        "total_entities": 0,
        "total_relationships": 0,
    }


def _log_planned_graph_tool_calls(tool_calls: list[dict], round_index: int | None = None) -> None:
    for index, tool_call in enumerate(tool_calls, start=1):
        arguments = tool_call["arguments"]
        logger.info(
            "graph_tool_planning:planned count=%s round=%s index=%s tool=%s node_id=%s node_types=%s relationship_types=%s page=%s",
            len(tool_calls),
            round_index,
            index,
            tool_call["tool_name"],
            arguments.get("node_id", ""),
            arguments.get("node_types", []),
            arguments.get("relationship_types", []),
            arguments.get("page", 1),
        )


def _log_done_graph_tool_calls(tool_calls: list[dict], round_context: dict) -> None:
    for tool_call in tool_calls:
        arguments = tool_call["arguments"]
        node_id = arguments.get("node_id", "")
        lookup_note = _get_lookup_note(round_context, node_id)
        queried_node_type = _get_queried_node_type(round_context, node_id) or lookup_note.get(
            "resolved_entity_type"
        )
        logger.info(
            "graph_tool_call:done tool=%s node_id=%s queried_node_type=%s node_types=%s relationship_types=%s page=%s entities=%s relationships=%s lookup_status=%s resolved_node_id=%s",
            tool_call["tool_name"],
            node_id,
            queried_node_type,
            arguments.get("node_types", []),
            arguments.get("relationship_types", []),
            arguments.get("page", 1),
            len(round_context.get("entities", [])),
            len(round_context.get("relationships", [])),
            lookup_note.get("status"),
            lookup_note.get("resolved_node_id"),
        )


def _get_queried_node_type(context: dict, node_id: str) -> str | None:
    for entity in context.get("entities", []):
        if entity.get("id") == node_id:
            return entity.get("node_type") or entity.get("entity_type")
    return None


def _get_lookup_note(context: dict, node_id: str) -> dict:
    for note in context.get("lookup_notes", []):
        if note.get("requested_node_id") == node_id:
            return note
    return {}


def _merge_previous_and_current_graph_context(state: AgentState, current_graph_context: dict) -> dict:
    if state.get("iteration", 0) <= 0:
        return current_graph_context
    previous_graph_context = state.get("graph_context") or {}
    if not _has_graph_context(previous_graph_context):
        return current_graph_context
    return _merge_graph_contexts([previous_graph_context, current_graph_context])


def _merge_previous_and_current_tool_trace(state: AgentState, current_tool_trace: list[dict]) -> list[dict]:
    if state.get("iteration", 0) <= 0:
        return current_tool_trace
    return [*(state.get("tool_trace") or []), *current_tool_trace]


def _has_graph_context(graph_context: dict) -> bool:
    return bool(
        graph_context.get("entities")
        or graph_context.get("relationships")
        or graph_context.get("lookup_notes")
    )


def _format_graph_context_human_readable(graph_context: dict) -> str:
    entities = graph_context.get("entities", [])
    relationships = graph_context.get("relationships", [])
    lookup_notes = graph_context.get("lookup_notes", [])
    lines = []
    if not entities and not relationships and not lookup_notes:
        return "  (no graph context available)"
    if lookup_notes:
        lines.append(f"  Lookup notes ({len(lookup_notes)}):")
        for note in lookup_notes:
            status = note.get("status", "?")
            requested = note.get("requested_node_id", "?")
            resolved = note.get("resolved_node_id")
            if resolved:
                lines.append(f"    - {status}: {requested} -> {resolved}")
            else:
                lines.append(f"    - {status}: {requested}")
    lines.append(f"  Entities ({len(entities)}):")
    for entity in entities:
        eid = entity.get("id", "?")
        if entity.get("abstract") is not None:
            abstract = entity["abstract"]
            abstract_preview = abstract[:120] + "..." if len(abstract) > 120 else abstract
            lines.append(f"    - {eid} (Paper): {abstract_preview}")
        elif entity.get("description") is not None:
            description = entity["description"]
            desc_preview = description[:120] + "..." if len(description) > 120 else description
            etype = entity.get("entity_type", "")
            type_tag = f" ({etype})" if etype else ""
            lines.append(f"    - {eid}{type_tag}: {desc_preview}")
        else:
            etype = entity.get("entity_type", "")
            type_tag = f" ({etype})" if etype else ""
            lines.append(f"    - {eid}{type_tag}")
    lines.append(f"  Relationships ({len(relationships)}):")
    for rel in relationships:
        source = rel.get("source_id", "?")
        target = rel.get("target_id", "?")
        rtype = rel.get("relationship_type", "?")
        lines.append(f"    - {source} -[{rtype}]-> {target}")
    return "\n".join(lines)


def _format_tool_trace_human_readable(tool_trace: list[dict]) -> str:
    if not tool_trace:
        return "  (no previous graph tool trace)"
    lines = []
    for trace_entry in tool_trace:
        round_index = trace_entry.get("round", "?")
        observation = trace_entry.get("observation") or {}
        entity_count = len(observation.get("entities", []))
        relationship_count = len(observation.get("relationships", []))
        lines.append(
            f"  Round {round_index}: entities={entity_count}, relationships={relationship_count}"
        )
        for tool_call in trace_entry.get("tool_calls", []):
            tool_name = tool_call.get("tool_name", "?")
            arguments = json.dumps(
                tool_call.get("arguments", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(f"    - {tool_name} arguments={arguments}")
    return "\n".join(lines)


def _merge_graph_contexts(contexts: list[dict]) -> dict:
    if not contexts:
        return _empty_graph_context()
    entities = []
    relationships = []
    lookup_notes = []
    seen_entities = set()
    seen_relationships = set()
    seen_lookup_notes = set()
    total_entities = 0
    total_relationships = 0
    for context in contexts:
        total_entities += int(context.get("total_entities", 0) or 0)
        total_relationships += int(context.get("total_relationships", 0) or 0)
        for note in context.get("lookup_notes", []):
            key = (
                note.get("status"),
                note.get("requested_node_id"),
                note.get("resolved_node_id"),
            )
            if key in seen_lookup_notes:
                continue
            seen_lookup_notes.add(key)
            lookup_notes.append(note)
        for entity in context.get("entities", []):
            entity_id = entity.get("id")
            if not entity_id or entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            entities.append(entity)
        for relationship in context.get("relationships", []):
            key = (
                relationship.get("source_id"),
                relationship.get("target_id"),
                relationship.get("relationship_type"),
            )
            if key in seen_relationships:
                continue
            seen_relationships.add(key)
            relationships.append(relationship)
    merged = {
        "entities": entities,
        "relationships": relationships,
        "page": 1,
        "page_size": 20,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
    }
    if lookup_notes:
        merged["lookup_notes"] = lookup_notes
    return merged


def _is_node_in_list(node_id: str, allowed_list: list[str]) -> bool:
    target = node_id.strip().lower()
    for allowed in allowed_list:
        if allowed.strip().lower() == target:
            return True
    return False
