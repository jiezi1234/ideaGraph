from __future__ import annotations

PROMPT = """\
Extract search keywords for scientific literature retrieval.
Return JSON only with fields: keywords and retrieval_query.
keywords must be a list of 3 to 8 concise domain terms or phrases.
retrieval_query must be a short space-separated query built from the keywords.
Prefer technical concepts, methods, domains, tasks, and constraints.
Do not include explanation.
"""
