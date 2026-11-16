from __future__ import annotations

from src.agent.state import AgentStateUpdate, PaperSelectionState
from src.retrieval.selector import TopScorePaperSelector


def paper_selection(
    state: PaperSelectionState,
    selector: TopScorePaperSelector | None = None,
) -> AgentStateUpdate:
    active_selector = selector or TopScorePaperSelector()
    selected_paper = active_selector.select(
        state["user_query"],
        state["candidate_papers"],
    )
    return {"selected_paper": selected_paper}
