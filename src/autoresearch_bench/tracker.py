"""Result tracker for benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from autoresearch_bench.extractor import Technique
from autoresearch_bench.papers import Paper


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    baseline_metric: float
    modified_metric: float
    improvement_pct: float
    hypothesis: str
    technique_name: str


class ResultTracker:
    """Track benchmark results with paper attribution."""

    def __init__(self) -> None:
        self._results: List[dict] = []

    def log_result(
        self, paper: Paper, technique: Technique, result: BenchmarkResult
    ) -> None:
        """Log a benchmark result."""
        self._results.append(
            {
                "paper_title": paper.title,
                "paper_url": paper.url,
                "technique_name": result.technique_name,
                "technique_category": technique.category,
                "baseline_metric": result.baseline_metric,
                "modified_metric": result.modified_metric,
                "improvement_pct": result.improvement_pct,
                "hypothesis": result.hypothesis,
            }
        )

    def get_results(self) -> List[dict]:
        """Get all tracked results."""
        return list(self._results)

    def find_best_techniques(self, top_n: int = 5) -> List[dict]:
        """Find the top N techniques by improvement percentage."""
        if not self._results:
            return []
        sorted_results = sorted(
            self._results, key=lambda r: r["improvement_pct"], reverse=True
        )
        return sorted_results[:top_n]

    def export_markdown(self) -> str:
        """Export results as markdown report."""
        if not self._results:
            return "# Benchmark Results\n\nNo results recorded yet.\n"

        lines = ["# Benchmark Results\n"]
        lines.append("| Technique | Category | Baseline | Modified | Improvement | Paper |")
        lines.append("|-----------|----------|----------|----------|-------------|-------|")

        for r in sorted(self._results, key=lambda x: x["improvement_pct"], reverse=True):
            lines.append(
                f"| {r['technique_name']} | {r['technique_category']} | "
                f"{r['baseline_metric']:.4f} | {r['modified_metric']:.4f} | "
                f"{r['improvement_pct']:.2f}% | {r['paper_title']} |"
            )
        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        """Export results as JSON."""
        return json.dumps(self._results, indent=2)
