"""Tests for benchmark runner."""

import pytest
from autoresearch_bench.runner import BenchmarkRunner


class TestBenchmarkRunner:
    def setup_method(self):
        self.runner = BenchmarkRunner()

    def test_compute_improvement(self):
        improvement = self.runner.compute_improvement(0.80, 0.85)
        assert abs(improvement - 6.25) < 0.01

    def test_compute_improvement_negative(self):
        improvement = self.runner.compute_improvement(0.80, 0.75)
        assert improvement < 0

    def test_compute_improvement_zero_baseline(self):
        improvement = self.runner.compute_improvement(0.0, 0.5)
        assert improvement == float("inf") or improvement > 100

    def test_create_benchmark_result(self):
        result = self.runner.create_result(
            baseline_metric=0.80,
            modified_metric=0.85,
            hypothesis="Adam is better",
            technique_name="Adam",
        )
        assert result.baseline_metric == 0.80
        assert result.modified_metric == 0.85
        assert result.improvement_pct == pytest.approx(6.25, rel=0.01)
        assert result.technique_name == "Adam"
