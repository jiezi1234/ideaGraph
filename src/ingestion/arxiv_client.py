from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx


logger = logging.getLogger(__name__)
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    def __init__(
        self,
        base_url: str = "https://export.arxiv.org/api/query",
        timeout: float = 20.0,
        http_client=None,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._http_client = http_client

    def find_pdf_by_title(self, title: str) -> str | None:
        normalized_title = " ".join(title.split())
        if not normalized_title:
            return None
        params = {
            "search_query": f'ti:"{normalized_title}"',
            "start": 0,
            "max_results": 1,
        }
        try:
            if self._http_client is not None:
                response = self._http_client.get(self._base_url, params=params)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(self._base_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("arxiv:query_failed title=%s error=%s message=%s", normalized_title, type(exc).__name__, exc)
            return None

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            logger.warning("arxiv:parse_failed title=%s message=%s", normalized_title, exc)
            return None

        entry = root.find("atom:entry", _ATOM_NS)
        if entry is None:
            return None
        for link in entry.findall("atom:link", _ATOM_NS):
            if link.attrib.get("type") == "application/pdf" and link.attrib.get("href"):
                return link.attrib["href"]
        arxiv_id = entry.findtext("atom:id", default="", namespaces=_ATOM_NS).rstrip("/")
        if "/abs/" in arxiv_id:
            return arxiv_id.replace("/abs/", "/pdf/")
        return None
