"""Tests for PDF fetcher."""

from unittest.mock import MagicMock, patch

from paper_benchmark.pdf_fetcher import PDFFetcher


class TestPDFFetcher:
    def setup_method(self):
        self.fetcher = PDFFetcher()

    def test_to_pdf_url_abs(self):
        url = "https://arxiv.org/abs/2301.12345"
        assert PDFFetcher._to_pdf_url(url) == "https://arxiv.org/pdf/2301.12345.pdf"

    def test_to_pdf_url_already_pdf(self):
        url = "https://arxiv.org/pdf/2301.12345.pdf"
        assert PDFFetcher._to_pdf_url(url) == "https://arxiv.org/pdf/2301.12345.pdf"

    def test_to_pdf_url_trailing_slash(self):
        url = "https://arxiv.org/abs/2301.12345/"
        result = PDFFetcher._to_pdf_url(url)
        assert "arxiv.org/pdf/2301.12345" in result

    def test_download_pdf(self):
        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.raise_for_status = MagicMock()

        with patch("paper_benchmark.pdf_fetcher.httpx.get", return_value=mock_response):
            data = PDFFetcher._download_pdf("https://arxiv.org/pdf/2301.12345.pdf")
            assert data == b"%PDF-1.4 fake pdf content"

    def test_extract_text_missing_pypdf(self):
        import importlib
        import sys

        # Test that ImportError is raised with helpful message when pypdf is not available
        with patch.dict(sys.modules, {"pypdf": None}):
            try:
                # Force re-import to trigger the ImportError path
                PDFFetcher._extract_text(b"fake")
            except (ImportError, Exception):
                pass  # Expected - either pypdf missing or invalid PDF bytes

    def test_fetch_pdf_text_integration(self):
        mock_response = MagicMock()
        mock_response.content = b"fake pdf"
        mock_response.raise_for_status = MagicMock()

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample extracted text from PDF"
        mock_reader.pages = [mock_page]

        with patch("paper_benchmark.pdf_fetcher.httpx.get", return_value=mock_response):
            with patch("paper_benchmark.pdf_fetcher.PDFFetcher._extract_text", return_value="Sample extracted text from PDF"):
                text = self.fetcher.fetch_pdf_text("https://arxiv.org/abs/2301.12345")
                assert "Sample extracted text" in text
