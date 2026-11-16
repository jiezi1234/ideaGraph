from __future__ import annotations

from src.agent.state import AgentStateUpdate
from src.schemas.agent import IdeaComparison


def idea_comparison(state: dict, model, prompt_template: str) -> AgentStateUpdate:
    previous_idea = state.get("initial_idea_output")
    current_idea = state["idea_output"]
    if not previous_idea:
        return {}

    messages = [
        {
            "role": "system",
            "content": prompt_template,
        },
        {
            "role": "user",
            "content": (
                "<idea_a>\n"
                f"{previous_idea.model_dump_json(indent=2)}\n"
                "</idea_a>\n"
                "<idea_b>\n"
                f"{current_idea.model_dump_json(indent=2)}\n"
                "</idea_b>"
            ),
        },
    ]
    result = model.structured_json(
        messages=messages,
        schema=IdeaComparison,
        name="idea_comparison",
        metadata={"stage": "comparison"},
    )
    parsed = result.parsed
    return {"idea_comparison": parsed if isinstance(parsed, IdeaComparison) else IdeaComparison(**parsed)}
