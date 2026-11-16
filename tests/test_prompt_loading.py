from pathlib import Path

from src.app.bootstrap import (
    load_chat_idea_agent_prompt_template,
    load_convert_paper_to_idea_prompt_template,
    load_evaluate_idea_prompt_template,
    load_extract_paper_abstract_prompt_template,
    load_extract_prompt_template,
    load_understand_query_prompt_template,
)
from src.prompts import IMPROVE_IDEA_AGENT_PROMPT


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "prompts"


def test_prompt_templates_are_python_modules_not_text_files():
    text_prompts = sorted(path.name for path in PROMPTS_DIR.glob("*.txt"))
    python_prompts = sorted(path.name for path in PROMPTS_DIR.glob("*.py"))

    assert text_prompts == []
    assert {
        "chat_idea_agent.py",
        "convert_paper_to_idea.py",
        "evaluate_research_idea.py",
        "extract_paper_abstract.py",
        "extract_paper_entities.py",
        "improve_idea_agent.py",
        "understand_query.py",
    }.issubset(set(python_prompts))


def test_extract_prompt_template_keeps_semantic_rules_without_inline_json_schema():
    prompt = load_extract_prompt_template()

    assert "Concept" in prompt
    assert "Domain" in prompt
    assert "Problem" in prompt
    assert "Method" in prompt
    assert "Scenario" in prompt
    assert "Innovation" in prompt
    assert "Used_For" in prompt
    assert "Contributes_To" in prompt
    assert "Use only information directly supported" in prompt
    assert "Do not invent entities" in prompt
    assert "Canonical entity naming" in prompt
    assert "canonical lowercase noun phrase" in prompt
    assert "Do not encode claims, relations, or evidence snippets in entity_id" in prompt
    assert "Entity naming examples" in prompt
    assert "Good entity_ids: \"bus scheduling\", \"multi-route\"" in prompt
    assert "Bad entity_ids: \"multi_route_bus_scheduling_problem\"" in prompt
    assert "Good entity_id: \"transformer decoder detection head\"" in prompt
    assert "Bad entity_id: \"transformer_decoder_detection_head\"" in prompt
    assert "Do not replace spaces with underscores" in prompt
    assert "must be factually supported by the provided title, abstract, or full text" in prompt
    assert "Define the entity itself as an independent scientific or engineering object" in prompt
    assert "express them with" in prompt
    assert "multi-route bus scheduling problem" in prompt
    assert "Problem entity \"bus scheduling\"" in prompt
    assert "separate Scenario or Concept entity" in prompt
    assert "an entity is constrained by or depends on" in prompt
    assert "Cross-paper mergeability check" in prompt
    assert "paper-internal architectural positions" in prompt
    assert "first transformer decoder layer" in prompt
    assert "encode them in relation descriptions" in prompt
    assert "current entity/relation set" in prompt
    assert "compact [field]----[field] line format" in prompt
    assert "structured schema" in prompt
    assert "Return only changed entities and relations" in prompt
    assert "Do not return unchanged entries" in prompt
    assert "Do not write vague descriptions" not in prompt
    assert "Do not add performance claims" not in prompt
    assert "JSON schema" not in prompt
    assert "Example output" not in prompt


def test_extract_abstract_prompt_keeps_semantic_rules_without_inline_output_schema():
    prompt = load_extract_paper_abstract_prompt_template()

    assert "If the paper text contains an abstract or 摘要 section" in prompt
    assert "reproduce its content as faithfully as possible" in prompt
    assert "Do not invent claims" in prompt
    assert "Output schema" not in prompt


def test_evaluate_idea_prompt_template_contains_rubric_without_inline_json_schema():
    prompt = load_evaluate_idea_prompt_template()

    assert "<role>" in prompt
    assert "<task>" in prompt
    assert "<context>" in prompt
    assert "<tool_strategy>" not in prompt
    assert "<evaluation_criteria>" in prompt
    assert "<output_requirements>" in prompt
    assert "Evaluation Agent" in prompt
    assert "Novelty" in prompt
    assert "Feasibility" in prompt
    assert "Expected Effectiveness" in prompt
    assert "Return the structured evaluation fields" in prompt
    assert "JSON schema" not in prompt
    assert "Reference rubric" not in prompt


def test_convert_paper_to_idea_prompt_keeps_task_rules_without_inline_schema():
    prompt = load_convert_paper_to_idea_prompt_template()

    assert "研究想法转换助手" in prompt
    assert "所有面向用户的字段必须使用中文" in prompt
    assert "core_paper_id 必须使用输入中的论文 id" in prompt
    assert "不要包含 references" in prompt
    assert "<schema>" not in prompt
    assert "只输出 JSON" not in prompt
    assert "IdeaOutput schema" not in prompt


def test_understand_query_prompt_template_contains_chat_idea_contract():
    prompt = load_understand_query_prompt_template()

    assert "Extract search keywords" in prompt
    assert "keywords" in prompt
    assert "retrieval_query" in prompt
    assert "Return JSON only" in prompt


def test_chat_idea_agent_prompt_template_contains_tool_and_generation_contracts():
    prompt = load_chat_idea_agent_prompt_template()
    normalized_prompt = " ".join(prompt.split())

    assert "<role>" in prompt
    assert "<task>" in prompt
    assert "<tool_use>" in prompt
    assert "<generation_output>" in prompt
    assert "expert AI researcher" in prompt
    assert "get_node_neighborhood" in prompt
    assert "up to 10 action rounds" in prompt
    assert "Batch independent calls" in prompt
    assert "Explore graph context with deliberate effort before finalizing" in prompt
    assert "Do not finalize merely because one graph observation exists" in prompt
    assert "promising papers, entities, relationships, gaps, methods, scenarios, or evaluation signals" in prompt
    assert "chain-of-evidence exploration" in prompt
    assert "paper -> problem or limitation -> alternative method or paper" in prompt
    assert "same problem or limitation" in prompt
    assert "<innovation_style>" in prompt
    assert "deeply think through and explore graph content" in prompt.lower()
    assert "cross-domain" in prompt
    assert "entities that are not already directly connected" in normalized_prompt
    assert "There is no generate_idea tool" in prompt
    assert "do not call any tool" in prompt
    assert "normal assistant response" in prompt
    assert "IdeaOutput JSON" in prompt
    assert "invalid content falls back" in prompt
    assert "Use Chinese" in prompt
    assert "Return JSON only" in prompt
    assert "title" in prompt
    assert "problem" in prompt
    assert "method" in prompt
    assert "experiment" in prompt.lower()
    assert "core_paper_title" in prompt
    assert "idea_lineage" in prompt
    assert "idea's development path" in prompt
    assert "Do not describe hidden model reasoning" in prompt
    assert "产生脉络" not in prompt
    assert "Problem" in prompt
    assert "Method" in prompt
    assert "Experiment" in prompt
    assert "related papers" in prompt
    assert "related entity ids" in prompt
    assert "entities" in prompt
    assert "Problem Statement" in prompt
    assert "Motivation" in prompt
    assert "Proposed Method" in prompt
    assert "Step-by-Step Experiment Plan" in prompt
    assert "Test Case Examples" in prompt
    assert "Fallback Plan" in prompt
    assert "JSON string" in prompt
    assert '{"tool_calls":[]}' not in prompt


def test_improve_idea_agent_prompt_encourages_deliberate_graph_exploration():
    prompt = IMPROVE_IDEA_AGENT_PROMPT
    normalized_prompt = " ".join(prompt.split())

    assert "Explore graph context with deliberate effort before finalizing" in prompt
    assert "Do not finalize merely because one graph observation exists" in prompt
    assert "promising papers, entities, relationships, gaps, methods, scenarios, or evaluation signals" in prompt
    assert "chain-of-evidence exploration" in prompt
    assert "paper -> problem or limitation -> alternative method or paper" in prompt
    assert "<innovation_style>" in prompt
    assert "deeply think through and explore graph content" in prompt.lower()
    assert "cross-domain" in prompt
    assert "entities that are not already directly connected" in normalized_prompt
    assert "idea_lineage" in prompt
