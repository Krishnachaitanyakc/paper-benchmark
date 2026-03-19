"""Tests for result tracker."""

import json
import pytest
from autoresearch_bench.tracker import ResultTracker, BenchmarkResult
from autoresearch_bench.papers import Paper
from autoresearch_bench.extractor import Technique


class TestBenchmarkResult:
    def test_creation(self):
        r = BenchmarkResult(
            baseline_metric=0.80,
            modified_metric=0.85,
            improvement_pct=6.25,
            hypothesis="Adam converges faster",
            technique_name="Adam optimizer",
        )
        assert r.improvement_pct == 6.25


class TestResultTracker:
    def setup_method(self):
        self.tracker = ResultTracker()

    def test_log_result(self):
        paper = Paper("Test Paper", "Abstract", "https://example.com", [])
        technique = Technique("dropout", "desc", "regularization", "hint")
        result = BenchmarkResult(0.80, 0.85, 6.25, "hypothesis", "dropout")

        self.tracker.log_result(paper, technique, result)
        results = self.tracker.get_results()
        assert len(results) == 1
        assert results[0]["paper_title"] == "Test Paper"
        assert results[0]["technique_name"] == "dropout"
        assert results[0]["improvement_pct"] == 6.25

    def test_log_multiple_results(self):
        for i in range(3):
            paper = Paper(f"Paper {i}", "Abstract", "url", [])
            technique = Technique(f"tech_{i}", "desc", "optimizer", "hint")
            result = BenchmarkResult(0.80, 0.80 + i * 0.01, i, "hyp", f"tech_{i}")
            self.tracker.log_result(paper, technique, result)

        assert len(self.tracker.get_results()) == 3

    def test_find_best_techniques(self):
        for i, name in enumerate(["dropout", "adam", "warmup"]):
            paper = Paper(f"Paper {i}", "Abstract", "url", [])
            technique = Technique(name, "desc", "optimizer", "hint")
            result = BenchmarkResult(0.80, 0.80 + (i + 1) * 0.02, (i + 1) * 2.5, "hyp", name)
            self.tracker.log_result(paper, technique, result)

        best = self.tracker.find_best_techniques(top_n=2)
        assert len(best) == 2
        assert best[0]["technique_name"] == "warmup"  # highest improvement

    def test_export_markdown(self):
        paper = Paper("Test Paper", "Abstract", "https://example.com", [])
        technique = Technique("dropout", "desc", "regularization", "hint")
        result = BenchmarkResult(0.80, 0.85, 6.25, "hypothesis", "dropout")
        self.tracker.log_result(paper, technique, result)

        md = self.tracker.export_markdown()
        assert "# Benchmark Results" in md
        assert "dropout" in md
        assert "Test Paper" in md

    def test_empty_tracker(self):
        assert self.tracker.get_results() == []
        assert self.tracker.find_best_techniques() == []
        md = self.tracker.export_markdown()
        assert "No results" in md

    def test_export_json(self):
        paper = Paper("Test Paper", "Abstract", "url", [])
        technique = Technique("dropout", "desc", "regularization", "hint")
        result = BenchmarkResult(0.80, 0.85, 6.25, "hyp", "dropout")
        self.tracker.log_result(paper, technique, result)

        json_str = self.tracker.export_json()
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["technique_name"] == "dropout"
