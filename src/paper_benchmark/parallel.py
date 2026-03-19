"""Parallel benchmarking using concurrent.futures."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional

from paper_benchmark.runner import BenchmarkResult, BenchmarkRunner


@dataclass
class BenchmarkTask:
    """A single benchmark task to run."""

    technique_name: str
    hypothesis: str
    run_fn: Callable[[], tuple]  # returns (baseline_metric, modified_metric)


class ParallelBenchmarkRunner:
    """Run multiple benchmarks in parallel."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._runner = BenchmarkRunner()

    def run_parallel(self, tasks: List[BenchmarkTask]) -> List[BenchmarkResult]:
        """Run benchmark tasks in parallel and collect results."""
        results: List[BenchmarkResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(task.run_fn): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    baseline, modified = future.result()
                    result = self._runner.create_result(
                        baseline_metric=baseline,
                        modified_metric=modified,
                        hypothesis=task.hypothesis,
                        technique_name=task.technique_name,
                    )
                    results.append(result)
                except Exception:
                    # Record failed benchmark with zero improvement
                    results.append(
                        BenchmarkResult(
                            baseline_metric=0.0,
                            modified_metric=0.0,
                            improvement_pct=0.0,
                            hypothesis=task.hypothesis,
                            technique_name=task.technique_name,
                        )
                    )

        return results
