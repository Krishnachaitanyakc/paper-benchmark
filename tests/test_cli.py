"""Tests for CLI."""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from autoresearch_bench.cli import cli


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_fetch_command(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        with patch("autoresearch_bench.papers.httpx.get", return_value=mock_response):
            result = self.runner.invoke(cli, ["fetch", "test query"])
            assert result.exit_code == 0

    def test_extract_command(self):
        result = self.runner.invoke(
            cli,
            ["extract", "We use dropout regularization and Adam optimizer."],
        )
        assert result.exit_code == 0

    def test_results_command(self):
        result = self.runner.invoke(cli, ["results"])
        assert result.exit_code == 0

    def test_report_command(self):
        result = self.runner.invoke(cli, ["report"])
        assert result.exit_code == 0
