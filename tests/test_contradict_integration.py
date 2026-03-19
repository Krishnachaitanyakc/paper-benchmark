"""Tests for contradict integration."""

from paper_benchmark.contradict_integration import ContradictIntegration, Contradiction


class TestContradictIntegration:
    def test_unavailable_when_no_module(self):
        ci = ContradictIntegration()
        assert ci.available is False

    def test_basic_contradiction_found(self):
        ci = ContradictIntegration()
        results = [
            {"technique_name": "dropout", "improvement_pct": 5.0},
            {"technique_name": "dropout", "improvement_pct": -2.0},
        ]
        contradictions = ci.detect_contradictions(results)
        assert len(contradictions) == 1
        assert contradictions[0].technique_a == "dropout"
        assert contradictions[0].severity == "medium"

    def test_no_contradiction(self):
        ci = ContradictIntegration()
        results = [
            {"technique_name": "dropout", "improvement_pct": 5.0},
            {"technique_name": "dropout", "improvement_pct": 3.0},
        ]
        contradictions = ci.detect_contradictions(results)
        assert len(contradictions) == 0

    def test_empty_results(self):
        ci = ContradictIntegration()
        contradictions = ci.detect_contradictions([])
        assert contradictions == []

    def test_multiple_techniques(self):
        ci = ContradictIntegration()
        results = [
            {"technique_name": "dropout", "improvement_pct": 5.0},
            {"technique_name": "dropout", "improvement_pct": -1.0},
            {"technique_name": "adam", "improvement_pct": 3.0},
            {"technique_name": "adam", "improvement_pct": 2.0},
        ]
        contradictions = ci.detect_contradictions(results)
        assert len(contradictions) == 1
        assert contradictions[0].technique_a == "dropout"

    def test_contradiction_dataclass(self):
        c = Contradiction(
            technique_a="a",
            technique_b="b",
            description="conflict",
            severity="high",
        )
        assert c.technique_a == "a"
        assert c.severity == "high"
