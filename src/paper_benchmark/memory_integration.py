"""Optional integration with autoresearch-memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from paper_benchmark.tracker import BenchmarkResult


class MemoryIntegration:
    """Store benchmark results in autoresearch-memory if available."""

    def __init__(self) -> None:
        self._memory_store: Optional[Any] = None
        self._available = False
        try:
            from autoresearch_memory import MemoryStore

            self._memory_store = MemoryStore()
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def store_result(self, result: BenchmarkResult, metadata: Optional[Dict] = None) -> bool:
        """Store a benchmark result in memory. Returns True on success."""
        if not self._available or self._memory_store is None:
            return False

        entry = {
            "type": "benchmark_result",
            "technique": result.technique_name,
            "baseline": result.baseline_metric,
            "modified": result.modified_metric,
            "improvement_pct": result.improvement_pct,
            "hypothesis": result.hypothesis,
        }
        if metadata:
            entry.update(metadata)

        try:
            self._memory_store.store(entry)
            return True
        except Exception:
            return False

    def retrieve_results(self, technique_name: Optional[str] = None) -> List[Dict]:
        """Retrieve stored benchmark results from memory."""
        if not self._available or self._memory_store is None:
            return []

        try:
            query = {"type": "benchmark_result"}
            if technique_name:
                query["technique"] = technique_name
            return self._memory_store.query(query)
        except Exception:
            return []
