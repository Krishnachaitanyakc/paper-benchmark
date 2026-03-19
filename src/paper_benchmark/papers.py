"""Paper fetcher using Semantic Scholar API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import httpx


@dataclass
class Paper:
    """A research paper."""

    title: str
    abstract: str
    url: str
    techniques: List[str] = field(default_factory=list)


class PaperFetcher:
    """Fetch papers from Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def fetch_papers(self, query: str, limit: int = 10) -> List[Paper]:
        """Search for papers by keyword query."""
        try:
            response = httpx.get(
                self.BASE_URL,
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,url",
                },
                timeout=30.0,
            )
            if response.status_code != 200:
                return []

            data = response.json()
            papers = []
            for item in data.get("data", []):
                papers.append(
                    Paper(
                        title=item.get("title", ""),
                        abstract=item.get("abstract", "") or "",
                        url=item.get("url", ""),
                    )
                )
            return papers
        except Exception:
            return []
