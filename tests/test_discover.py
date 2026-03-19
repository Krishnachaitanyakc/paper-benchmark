"""Tests for paper discovery."""

from unittest.mock import MagicMock, patch

from paper_benchmark.discover import PaperDiscovery, RankedPaper


MOCK_DISCOVER_RESPONSE = {
    "data": [
        {
            "title": "Paper A",
            "abstract": "Abstract A",
            "url": "url_a",
            "citationCount": 100,
            "influentialCitationCount": 10,
            "year": 2023,
        },
        {
            "title": "Paper B",
            "abstract": "Abstract B",
            "url": "url_b",
            "citationCount": 50,
            "influentialCitationCount": 20,
            "year": 2022,
        },
    ]
}


class TestPaperDiscovery:
    def test_discover_returns_ranked(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_DISCOVER_RESPONSE

        with patch("paper_benchmark.discover.httpx.get", return_value=mock_resp):
            disc = PaperDiscovery()
            ranked = disc.discover("attention mechanism")
            assert len(ranked) == 2
            # Should be sorted by score descending
            assert ranked[0].score >= ranked[1].score

    def test_discover_empty(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}

        with patch("paper_benchmark.discover.httpx.get", return_value=mock_resp):
            disc = PaperDiscovery()
            ranked = disc.discover("nonexistent xyz")
            assert ranked == []

    def test_discover_api_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("paper_benchmark.discover.httpx.get", return_value=mock_resp):
            disc = PaperDiscovery()
            ranked = disc.discover("test")
            assert ranked == []

    def test_compute_score(self):
        score = PaperDiscovery._compute_score(100, 10, 2023)
        assert score > 0
        # Newer paper with same citations should score higher
        score_newer = PaperDiscovery._compute_score(100, 10, 2024)
        assert score_newer > score

    def test_compute_score_no_year(self):
        score = PaperDiscovery._compute_score(100, 10, None)
        assert score == 150.0  # 100 + 10*5

    def test_ranked_paper_fields(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_DISCOVER_RESPONSE

        with patch("paper_benchmark.discover.httpx.get", return_value=mock_resp):
            disc = PaperDiscovery()
            ranked = disc.discover("test")
            rp = ranked[0]
            assert isinstance(rp, RankedPaper)
            assert rp.paper.title != ""
            assert rp.citation_count >= 0
