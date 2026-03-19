"""Auto paper discovery via Semantic Scholar with citation ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx

from paper_benchmark.papers import Paper


@dataclass
class RankedPaper:
    """A paper with citation-based ranking information."""

    paper: Paper
    citation_count: int
    influential_citation_count: int
    year: Optional[int]
    score: float


class PaperDiscovery:
    """Discover and rank recent relevant papers using Semantic Scholar."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def discover(
        self,
        keywords: str,
        limit: int = 20,
        year_from: Optional[int] = None,
    ) -> List[RankedPaper]:
        """Search for papers and rank by citation growth."""
        params: dict = {
            "query": keywords,
            "limit": min(limit, 100),
            "fields": "title,abstract,url,citationCount,influentialCitationCount,year",
        }
        if year_from is not None:
            params["year"] = f"{year_from}-"

        try:
            response = httpx.get(self.BASE_URL, params=params, timeout=30.0)
            if response.status_code != 200:
                return []
            data = response.json()
        except Exception:
            return []

        ranked = []
        for item in data.get("data", []):
            citation_count = item.get("citationCount", 0) or 0
            influential = item.get("influentialCitationCount", 0) or 0
            year = item.get("year")
            score = self._compute_score(citation_count, influential, year)

            paper = Paper(
                title=item.get("title", ""),
                abstract=item.get("abstract", "") or "",
                url=item.get("url", ""),
            )
            ranked.append(
                RankedPaper(
                    paper=paper,
                    citation_count=citation_count,
                    influential_citation_count=influential,
                    year=year,
                    score=score,
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    @staticmethod
    def _compute_score(
        citations: int, influential: int, year: Optional[int]
    ) -> float:
        """Compute ranking score emphasizing recent, influential papers."""
        base = citations + influential * 5
        if year is not None:
            recency_bonus = max(0, year - 2018) * 2
            base += recency_bonus
        return float(base)
