"""Tests for LLM extractor."""

import json
from unittest.mock import MagicMock, patch

from paper_benchmark.llm_extractor import LLMExtractor


class TestLLMExtractor:
    def test_parse_response_valid(self):
        response = json.dumps([
            {
                "name": "dropout",
                "description": "Randomly drop units",
                "category": "regularization",
                "implementation_hint": "nn.Dropout(p=0.5)",
            }
        ])
        techniques = LLMExtractor._parse_response(response)
        assert len(techniques) == 1
        assert techniques[0].name == "dropout"
        assert techniques[0].category == "regularization"

    def test_parse_response_with_surrounding_text(self):
        response = 'Here are the techniques:\n[{"name": "adam", "description": "optimizer", "category": "optimizer", "implementation_hint": "use Adam"}]\nDone.'
        techniques = LLMExtractor._parse_response(response)
        assert len(techniques) == 1
        assert techniques[0].name == "adam"

    def test_parse_response_no_json(self):
        try:
            LLMExtractor._parse_response("No techniques found in this text.")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_fallback_on_failure(self):
        extractor = LLMExtractor(api_key="fake-key")
        # _extract_with_llm will fail (no real API), should fall back to regex
        text = "We use dropout regularization to prevent overfitting in deep networks."
        techniques = extractor.extract_from_abstract(text)
        assert len(techniques) >= 1
        assert any(t.name == "dropout" for t in techniques)

    def test_extract_with_llm_mock(self):
        mock_content = MagicMock()
        mock_content.text = json.dumps([
            {
                "name": "mixup",
                "description": "interpolate samples",
                "category": "data_augmentation",
                "implementation_hint": "mix inputs",
            }
        ])
        mock_message = MagicMock()
        mock_message.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        import sys
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            extractor = LLMExtractor(api_key="fake")
            techniques = extractor._extract_with_llm("test abstract")
            assert len(techniques) == 1
            assert techniques[0].name == "mixup"
