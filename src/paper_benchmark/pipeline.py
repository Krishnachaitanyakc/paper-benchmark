"""End-to-end pipeline: fetch paper -> extract techniques -> generate code -> track results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from paper_benchmark.codegen import CodeGenerator
from paper_benchmark.extractor import Technique, TechniqueExtractor
from paper_benchmark.papers import Paper, PaperFetcher
from paper_benchmark.runner import BenchmarkRunner
from paper_benchmark.tracker import ResultTracker


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    paper: Paper
    techniques: List[Technique]
    modifications: List[str]
    results_logged: int


class Pipeline:
    """Chain: fetch paper -> extract techniques -> generate code -> track results."""

    def __init__(
        self,
        fetcher: Optional[PaperFetcher] = None,
        extractor: Optional[TechniqueExtractor] = None,
        codegen: Optional[CodeGenerator] = None,
        runner: Optional[BenchmarkRunner] = None,
        tracker: Optional[ResultTracker] = None,
    ):
        self.fetcher = fetcher or PaperFetcher()
        self.extractor = extractor or TechniqueExtractor()
        self.codegen = codegen or CodeGenerator()
        self.runner = runner or BenchmarkRunner()
        self.tracker = tracker or ResultTracker()

    def run(self, query: str, base_code: str, limit: int = 1) -> List[PipelineResult]:
        """Run the full pipeline for a search query."""
        papers = self.fetcher.fetch_papers(query, limit=limit)
        results = []
        for paper in papers:
            result = self.run_for_paper(paper, base_code)
            results.append(result)
        return results

    def run_for_paper(self, paper: Paper, base_code: str) -> PipelineResult:
        """Run pipeline for a single paper."""
        techniques = self.extractor.extract_from_abstract(paper.abstract)
        modifications = []
        logged = 0

        for technique in techniques:
            modified_code = self.codegen.generate_modification(technique, base_code)
            modifications.append(modified_code)

            bench_result = self.runner.create_result(
                baseline_metric=0.0,
                modified_metric=0.0,
                hypothesis=f"Applying {technique.name} from '{paper.title}'",
                technique_name=technique.name,
            )
            self.tracker.log_result(paper, technique, bench_result)
            logged += 1

        return PipelineResult(
            paper=paper,
            techniques=techniques,
            modifications=modifications,
            results_logged=logged,
        )
