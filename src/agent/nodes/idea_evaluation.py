from __future__ import annotations

import json
import logging

from src.agent.state import AgentStateUpdate, IdeaEvaluationState
from src.schemas.agent import IdeaEvaluation
from src.tools.chat_idea import create_chat_idea_tools

logger = logging.getLogger(__name__)

_MAX_EVALUATION_TOOL_ROUNDS = 5


def idea_evaluation(
    state: IdeaEvaluationState,
    model,
    prompt_template: str,
    graph_context_provider=None,
) -> AgentStateUpdate:
    messages = _build_evaluation_messages(state, prompt_template)
    if graph_context_provider is None:
        return _structured_evaluation(model=model, messages=messages)

    tools = create_chat_idea_tools(graph_context_provider)
    tool_by_name = {tool.name: tool for tool in tools}
    openai_tools = [tool.openai_schema() for tool in tools]
    round_contexts = []
    tool_trace = []

    for round_index in range(1, _MAX_EVALUATION_TOOL_ROUNDS + 1):
        response = model.chat_with_tools(
            messages=messages,
            tools=openai_tools,
            name=f"idea_evaluation_round_{round_index}",
            metadata={"stage": "evaluation_agent_round", "round": round_index},
        )
        messages.append(_assistant_message_from_response(response))
        tool_calls = _normalize_tool_calls(response.tool_calls)
        if not tool_calls:
            break

        tool_results = _execute_tool_calls(tool_by_name, response.tool_calls)
        for tool_call, tool_result in zip(response.tool_calls, tool_results, strict=True):
            messages.append(_tool_result_message(tool_call, tool_result))
        round_context = _merge_graph_contexts(tool_results)
        round_contexts.append(round_context)
        tool_trace.append(
            {
                "round": round_index,
                "tool_calls": tool_calls,
                "observation": round_context,
            }
        )
    else:
        logger.info("evaluation_agent:max_rounds_reached rounds=%s", _MAX_EVALUATION_TOOL_ROUNDS)

    evaluation_graph_context = _merge_graph_contexts(round_contexts)
    final_messages = [
        *messages,
        {
            "role": "system",
            "content": (
                "Current phase: final_idea_evaluation\n"
                "Return only the structured IdeaEvaluation fields. Use prior graph tool "
                "observations in this message history as explicit evidence when they are relevant."
            ),
        },
    ]
    update = _structured_evaluation(model=model, messages=final_messages)
    update["evaluation_graph_context"] = evaluation_graph_context
    update["evaluation_tool_trace"] = tool_trace
    return update


def _build_evaluation_messages(state: IdeaEvaluationState, prompt_template: str) -> list[dict]:
    idea_output = state["idea_output"]
    user_content = (
        "<generated_idea>\n"
        f"{idea_output.model_dump_json(indent=2)}\n"
        "</generated_idea>"
    )
    system_content = _build_evaluation_system_content(state, prompt_template)
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _build_evaluation_system_content(state: IdeaEvaluationState, prompt_template: str) -> str:
    system_content = prompt_template
    runtime_context = _format_available_runtime_context(state)
    if runtime_context:
        # 评估上下文属于评测 agent 的可见运行证据，放在 system 侧，避免污染用户任务输入。
        system_content += f"\n\n<available_runtime_context>\n{runtime_context}</available_runtime_context>"
    return system_content


def _format_available_runtime_context(state: IdeaEvaluationState) -> str:
    lines = []
    if state.get("user_query"):
        lines.append(f"User query: {state['user_query']}")
    selected_paper = state.get("selected_paper")
    if selected_paper is not None:
        lines.append(
            "Selected paper:\n"
            f"{json.dumps(_to_jsonable(selected_paper), ensure_ascii=False, indent=2)}"
        )
    candidate_papers = state.get("candidate_papers") or []
    if candidate_papers:
        lines.append(
            "Candidate papers:\n"
            f"{json.dumps(_to_jsonable(candidate_papers[:5]), ensure_ascii=False, indent=2)}"
        )
    if state.get("graph_context"):
        lines.append(
            "Generation graph context:\n"
            f"{json.dumps(_to_jsonable(state['graph_context']), ensure_ascii=False, indent=2)}"
        )
    return "\n".join(lines)


def _structured_evaluation(model, messages: list[dict]) -> AgentStateUpdate:
    result = model.structured_json(
        messages=messages,
        schema=IdeaEvaluation,
        name="idea_evaluation",
        metadata={"stage": "evaluation"},
    )
    parsed = result.parsed
    return {"idea_evaluation": parsed if isinstance(parsed, IdeaEvaluation) else IdeaEvaluation(**parsed)}


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


def _normalize_tool_calls(tool_calls: list) -> list[dict]:
    normalized = []
    for tool_call in tool_calls or []:
        args = dict(getattr(tool_call, "arguments", {}) or {})
        normalized.append(
            {
                "tool_name": getattr(tool_call, "tool_name", ""),
                "tool_call_id": getattr(tool_call, "tool_call_id", ""),
                "arguments": {
                    "node_id": args.get("node_id", ""),
                    "node_types": args.get("node_types") or [],
                    "relationship_types": args.get("relationship_types") or [],
                    "page": args.get("page") or 1,
                },
            }
        )
    return normalized


def _execute_tool_calls(tool_by_name: dict, tool_calls: list) -> list[dict]:
    results = []
    for tool_call in tool_calls:
        tool = tool_by_name.get(tool_call.tool_name)
        if tool is None:
            results.append(_tool_error_result(tool_call, ValueError(f"Unknown graph tool: {tool_call.tool_name}")))
            continue
        try:
            results.append(tool.call(tool_call.arguments))
        except Exception as error:
            # 评测阶段的工具失败也作为可恢复证据返回，避免整次评分直接崩溃。
            results.append(_tool_error_result(tool_call, error))
            logger.warning("evaluation_graph_tool_call:error tool=%s", tool_call.tool_name, exc_info=True)
    return results


def _tool_error_result(tool_call, error: Exception) -> dict:
    return {
        "status": "error",
        "tool_name": tool_call.tool_name,
        "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "recoverable": True,
        "entities": [],
        "relationships": [],
        "page": 1,
        "page_size": 20,
        "total_entities": 0,
        "total_relationships": 0,
    }


def _merge_graph_contexts(contexts: list[dict]) -> dict:
    merged_entities = {}
    merged_relationships = {}
    lookup_notes = []
    for context in contexts:
        for entity in context.get("entities", []) or []:
            entity_id = entity.get("id")
            if entity_id:
                merged_entities.setdefault(entity_id, entity)
        for relationship in context.get("relationships", []) or []:
            key = (
                relationship.get("source_id"),
                relationship.get("relationship_type"),
                relationship.get("target_id"),
            )
            merged_relationships.setdefault(key, relationship)
        lookup_notes.extend(context.get("lookup_notes", []) or [])
    return {
        "entities": list(merged_entities.values()),
        "relationships": list(merged_relationships.values()),
        "lookup_notes": lookup_notes,
        "page": 1,
        "page_size": 20,
        "total_entities": len(merged_entities),
        "total_relationships": len(merged_relationships),
    }


def _to_jsonable(value):
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value
