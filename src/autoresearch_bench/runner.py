"""Benchmark runner for comparing original vs modified code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a benchmark comparison."""

    baseline_metric: float
    modified_metric: float
    improvement_pct: float
    hypothesis: str
    technique_name: str


class BenchmarkRunner:
    """Run benchmarks comparing original vs modified code."""

    def compute_improvement(self, baseline: float, modified: float) -> float:
        """Compute percentage improvement from baseline to modified."""
        if baseline == 0:
            return float("inf")
        return ((modified - baseline) / baseline) * 100.0

    def create_result(
        self,
        baseline_metric: float,
        modified_metric: float,
        hypothesis: str,
        technique_name: str,
    ) -> BenchmarkResult:
        """Create a benchmark result with computed improvement."""
        improvement = self.compute_improvement(baseline_metric, modified_metric)
        return BenchmarkResult(
            baseline_metric=baseline_metric,
            modified_metric=modified_metric,
            improvement_pct=improvement,
            hypothesis=hypothesis,
            technique_name=technique_name,
        )
