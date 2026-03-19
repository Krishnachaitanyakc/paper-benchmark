"""Tests for memory integration."""

from unittest.mock import MagicMock, patch

from paper_benchmark.memory_integration import MemoryIntegration
from paper_benchmark.tracker import BenchmarkResult


class TestMemoryIntegration:
    def test_unavailable_when_no_module(self):
        mi = MemoryIntegration()
        # autoresearch_memory is not installed, should be unavailable
        assert mi.available is False

    def test_store_result_returns_false_when_unavailable(self):
        mi = MemoryIntegration()
        result = BenchmarkResult(0.8, 0.85, 6.25, "hyp", "dropout")
        assert mi.store_result(result) is False

    def test_retrieve_results_returns_empty_when_unavailable(self):
        mi = MemoryIntegration()
        assert mi.retrieve_results() == []

    def test_store_result_with_mock_memory(self):
        mi = MemoryIntegration()
        mock_store = MagicMock()
        mi._memory_store = mock_store
        mi._available = True

        result = BenchmarkResult(0.8, 0.85, 6.25, "hyp", "dropout")
        assert mi.store_result(result) is True
        mock_store.store.assert_called_once()

    def test_retrieve_results_with_mock_memory(self):
        mi = MemoryIntegration()
        mock_store = MagicMock()
        mock_store.query.return_value = [{"technique": "dropout"}]
        mi._memory_store = mock_store
        mi._available = True

        results = mi.retrieve_results("dropout")
        assert len(results) == 1

    def test_store_with_metadata(self):
        mi = MemoryIntegration()
        mock_store = MagicMock()
        mi._memory_store = mock_store
        mi._available = True

        result = BenchmarkResult(0.8, 0.85, 6.25, "hyp", "dropout")
        mi.store_result(result, metadata={"extra": "info"})
        call_args = mock_store.store.call_args[0][0]
        assert call_args["extra"] == "info"
