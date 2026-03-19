"""Tests for paper fetcher."""

import json
import pytest
from unittest.mock import patch, MagicMock
from paper_benchmark.papers import PaperFetcher, Paper


MOCK_SEARCH_RESPONSE = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Cosine Annealing for Deep Learning",
            "abstract": "We propose a cosine annealing learning rate schedule that improves convergence.",
            "url": "https://semanticscholar.org/paper/abc123",
        },
        {
            "paperId": "def456",
            "title": "Dropout as Regularization",
            "abstract": "We introduce dropout, a technique for regularization in neural networks.",
            "url": "https://semanticscholar.org/paper/def456",
        },
    ]
}


class TestPaper:
    def test_paper_creation(self):
        paper = Paper(
            title="Test Paper",
            abstract="This is a test abstract.",
            url="https://example.com/paper",
            techniques=["dropout"],
        )
        assert paper.title == "Test Paper"
        assert paper.abstract == "This is a test abstract."
        assert paper.url == "https://example.com/paper"
        assert paper.techniques == ["dropout"]


class TestPaperFetcher:
    def test_fetch_paper_returns_papers(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SEARCH_RESPONSE

        with patch("paper_benchmark.papers.httpx.get", return_value=mock_response):
            fetcher = PaperFetcher()
            papers = fetcher.fetch_papers("cosine annealing")
            assert len(papers) == 2
            assert papers[0].title == "Cosine Annealing for Deep Learning"

    def test_fetch_paper_empty_results(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("paper_benchmark.papers.httpx.get", return_value=mock_response):
            fetcher = PaperFetcher()
            papers = fetcher.fetch_papers("nonexistent topic xyz")
            assert papers == []

    def test_fetch_paper_api_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server Error")

        with patch("paper_benchmark.papers.httpx.get", return_value=mock_response):
            fetcher = PaperFetcher()
            papers = fetcher.fetch_papers("test")
            assert papers == []

    def test_extract_abstract(self):
        paper = Paper(
            title="Test",
            abstract="We propose dropout regularization.",
            url="https://example.com",
            techniques=[],
        )
        assert paper.abstract == "We propose dropout regularization."

    def test_extract_techniques_from_paper(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_SEARCH_RESPONSE

        with patch("paper_benchmark.papers.httpx.get", return_value=mock_response):
            fetcher = PaperFetcher()
            papers = fetcher.fetch_papers("cosine annealing")
            assert len(papers) > 0
