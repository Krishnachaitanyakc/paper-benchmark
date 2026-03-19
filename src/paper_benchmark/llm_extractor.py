"""LLM-based technique extraction using Anthropic Claude."""

from __future__ import annotations

import json
import re
from typing import List, Optional

from paper_benchmark.extractor import Technique, TechniqueExtractor

EXTRACTION_PROMPT = """Extract machine learning techniques from this paper abstract.
Return a JSON array where each element has:
- "name": technique name (lowercase)
- "description": one-sentence description from the text
- "category": one of "optimizer", "regularization", "architecture", "data_augmentation", "training_trick"
- "implementation_hint": brief PyTorch implementation hint

Return ONLY the JSON array, no other text.

Abstract:
{text}"""


class LLMExtractor:
    """Extract ML techniques using Claude API, with regex fallback."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self._api_key = api_key
        self._model = model
        self._fallback = TechniqueExtractor()

    def extract_from_abstract(self, text: str) -> List[Technique]:
        """Extract techniques using LLM, falling back to regex on failure."""
        try:
            return self._extract_with_llm(text)
        except Exception:
            return self._fallback.extract_from_abstract(text)

    def _extract_with_llm(self, text: str) -> List[Technique]:
        """Call Claude API for structured extraction."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for LLM extraction. Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}
            ],
        )

        response_text = message.content[0].text
        return self._parse_response(response_text)

    @staticmethod
    def _parse_response(response_text: str) -> List[Technique]:
        """Parse JSON response into Technique objects."""
        # Try to extract JSON array from response
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response")

        items = json.loads(match.group())
        techniques = []
        for item in items:
            techniques.append(
                Technique(
                    name=item["name"],
                    description=item.get("description", item["name"]),
                    category=item.get("category", "training_trick"),
                    implementation_hint=item.get("implementation_hint", ""),
                )
            )
        return techniques
