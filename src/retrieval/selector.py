from __future__ import annotations

from src.schemas.agent import CandidatePaper


class TopScorePaperSelector:
    def select(self, user_query: str, candidates: list[CandidatePaper]) -> CandidatePaper:
        del user_query
        if not candidates:
            raise ValueError("No candidate papers available for selection.")
        return max(candidates, key=lambda paper: paper.score)
