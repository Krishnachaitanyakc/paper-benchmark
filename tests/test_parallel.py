"""Tests for parallel benchmarking."""

from paper_benchmark.parallel import BenchmarkTask, ParallelBenchmarkRunner


class TestParallelBenchmarkRunner:
    def test_run_parallel_basic(self):
        runner = ParallelBenchmarkRunner(max_workers=2)
        tasks = [
            BenchmarkTask(
                technique_name="dropout",
                hypothesis="dropout helps",
                run_fn=lambda: (0.80, 0.85),
            ),
            BenchmarkTask(
                technique_name="adam",
                hypothesis="adam converges faster",
                run_fn=lambda: (0.80, 0.82),
            ),
        ]
        results = runner.run_parallel(tasks)
        assert len(results) == 2
        names = {r.technique_name for r in results}
        assert "dropout" in names
        assert "adam" in names

    def test_run_parallel_handles_failure(self):
        def failing_fn():
            raise RuntimeError("benchmark failed")

        runner = ParallelBenchmarkRunner(max_workers=1)
        tasks = [
            BenchmarkTask(
                technique_name="broken",
                hypothesis="this will fail",
                run_fn=failing_fn,
            ),
        ]
        results = runner.run_parallel(tasks)
        assert len(results) == 1
        assert results[0].improvement_pct == 0.0

    def test_run_parallel_empty(self):
        runner = ParallelBenchmarkRunner()
        results = runner.run_parallel([])
        assert results == []

    def test_run_parallel_computes_improvement(self):
        runner = ParallelBenchmarkRunner(max_workers=1)
        tasks = [
            BenchmarkTask(
                technique_name="test",
                hypothesis="test hypothesis",
                run_fn=lambda: (0.80, 0.88),
            ),
        ]
        results = runner.run_parallel(tasks)
        assert len(results) == 1
        assert abs(results[0].improvement_pct - 10.0) < 0.01
