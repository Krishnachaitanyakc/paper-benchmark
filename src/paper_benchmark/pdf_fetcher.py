"""Fetch and parse arXiv PDFs for technique extraction."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

import httpx


class PDFFetcher:
    """Download and extract text from arXiv PDFs."""

    def fetch_pdf_text(self, arxiv_url: str) -> str:
        """Download an arXiv PDF and extract text content."""
        pdf_url = self._to_pdf_url(arxiv_url)
        pdf_bytes = self._download_pdf(pdf_url)
        return self._extract_text(pdf_bytes)

    @staticmethod
    def _to_pdf_url(url: str) -> str:
        """Convert arXiv abstract URL to PDF URL."""
        # Handle various arXiv URL formats
        url = url.strip().rstrip("/")
        # arxiv.org/abs/XXXX.XXXXX -> arxiv.org/pdf/XXXX.XXXXX
        url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
        if not url.endswith(".pdf"):
            url += ".pdf"
        return url

    @staticmethod
    def _download_pdf(url: str) -> bytes:
        """Download PDF bytes from URL."""
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        return response.content

    @staticmethod
    def _extract_text(pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF parsing. Install with: pip install pypdf"
            )

        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
