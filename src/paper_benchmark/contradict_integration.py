"""Optional integration with autoresearch-contradict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Contradiction:
    """A detected contradiction between benchmark results."""

    technique_a: str
    technique_b: str
    description: str
    severity: str  # "low", "medium", "high"


class ContradictIntegration:
    """Detect contradictions in benchmark results using autoresearch-contradict."""

    def __init__(self) -> None:
        self._detector: Optional[Any] = None
        self._available = False
        try:
            from autoresearch_contradict import ContradictionDetector

            self._detector = ContradictionDetector()
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def detect_contradictions(self, results: List[Dict]) -> List[Contradiction]:
        """Run contradiction detection on benchmark results."""
        if not self._available or self._detector is None:
            return self._basic_contradiction_check(results)

        try:
            raw = self._detector.detect(results)
            return [
                Contradiction(
                    technique_a=c.get("technique_a", ""),
                    technique_b=c.get("technique_b", ""),
                    description=c.get("description", ""),
                    severity=c.get("severity", "low"),
                )
                for c in raw
            ]
        except Exception:
            return self._basic_contradiction_check(results)

    @staticmethod
    def _basic_contradiction_check(results: List[Dict]) -> List[Contradiction]:
        """Simple heuristic: flag techniques with conflicting improvement signs."""
        contradictions = []
        by_technique: Dict[str, List[Dict]] = {}
        for r in results:
            name = r.get("technique_name", "")
            by_technique.setdefault(name, []).append(r)

        for name, entries in by_technique.items():
            improvements = [e.get("improvement_pct", 0) for e in entries]
            has_positive = any(i > 0 for i in improvements)
            has_negative = any(i < 0 for i in improvements)
            if has_positive and has_negative:
                contradictions.append(
                    Contradiction(
                        technique_a=name,
                        technique_b=name,
                        description=f"'{name}' shows both positive and negative improvements across runs",
                        severity="medium",
                    )
                )

        return contradictions
