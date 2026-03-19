"""Tests for end-to-end pipeline."""

from unittest.mock import MagicMock, patch

from paper_benchmark.papers import Paper
from paper_benchmark.pipeline import Pipeline


class TestPipeline:
    def test_run_for_paper(self):
        paper = Paper(
            title="Test Paper",
            abstract="We use dropout regularization to prevent overfitting in deep models.",
            url="https://example.com",
        )
        pipe = Pipeline()
        result = pipe.run_for_paper(paper, "import torch\nmodel = nn.Sequential(nn.Linear(10,5), nn.ReLU())")
        assert result.paper.title == "Test Paper"
        assert len(result.techniques) >= 1
        assert result.results_logged == len(result.techniques)

    def test_run_for_paper_no_techniques(self):
        paper = Paper(
            title="History of Computing",
            abstract="This paper discusses the history of computing.",
            url="https://example.com",
        )
        pipe = Pipeline()
        result = pipe.run_for_paper(paper, "import torch")
        assert result.techniques == []
        assert result.results_logged == 0

    def test_run_with_mock_fetcher(self):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_papers.return_value = [
            Paper("Paper A", "We use dropout regularization to prevent overfitting.", "url1"),
        ]
        pipe = Pipeline(fetcher=mock_fetcher)
        results = pipe.run("dropout", "import torch", limit=1)
        assert len(results) == 1
        assert results[0].paper.title == "Paper A"

    def test_tracker_populated(self):
        paper = Paper(
            title="Test",
            abstract="We apply dropout regularization to prevent overfitting in networks.",
            url="url",
        )
        pipe = Pipeline()
        pipe.run_for_paper(paper, "import torch")
        tracked = pipe.tracker.get_results()
        assert len(tracked) >= 1
