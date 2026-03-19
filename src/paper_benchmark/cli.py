"""CLI for paper-benchmark."""

from __future__ import annotations

import click

from paper_benchmark.extractor import TechniqueExtractor
from paper_benchmark.papers import PaperFetcher
from paper_benchmark.tracker import ResultTracker

_tracker = ResultTracker()
_extractor = TechniqueExtractor()
_fetcher = PaperFetcher()


@click.group()
def cli() -> None:
    """paper-benchmark: Paper-to-benchmark pipeline."""


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of papers to fetch")
@click.option("--pdf", is_flag=True, help="Fetch arXiv PDF and extract text")
def fetch(query: str, limit: int, pdf: bool) -> None:
    """Search Semantic Scholar for papers."""
    if pdf:
        from paper_benchmark.pdf_fetcher import PDFFetcher

        pdf_fetcher = PDFFetcher()
        try:
            text = pdf_fetcher.fetch_pdf_text(query)
            click.echo(f"Extracted {len(text)} characters from PDF.")
            click.echo(text[:500] + "..." if len(text) > 500 else text)
        except Exception as e:
            click.echo(f"Error fetching PDF: {e}")
        return

    papers = _fetcher.fetch_papers(query, limit=limit)
    if not papers:
        click.echo("No papers found.")
        return
    for i, paper in enumerate(papers, 1):
        click.echo(f"{i}. {paper.title}")
        if paper.abstract:
            click.echo(f"   {paper.abstract[:100]}...")
        click.echo(f"   URL: {paper.url}")
        click.echo()


@cli.command()
@click.argument("text")
@click.option("--llm", is_flag=True, help="Use LLM-based extraction (requires anthropic SDK)")
def extract(text: str, llm: bool) -> None:
    """Extract ML techniques from text."""
    if llm:
        from paper_benchmark.llm_extractor import LLMExtractor

        extractor = LLMExtractor()
        techniques = extractor.extract_from_abstract(text)
    else:
        techniques = _extractor.extract_from_abstract(text)

    if not techniques:
        click.echo("No techniques found.")
        return
    for t in techniques:
        click.echo(f"- {t.name} [{t.category}]")
        click.echo(f"  Hint: {t.implementation_hint}")


@cli.command()
def results() -> None:
    """Show tracked benchmark results."""
    res = _tracker.get_results()
    if not res:
        click.echo("No benchmark results recorded yet.")
        return
    for r in res:
        click.echo(
            f"[{r['technique_name']}] {r['baseline_metric']:.4f} -> "
            f"{r['modified_metric']:.4f} ({r['improvement_pct']:+.2f}%)"
        )


@cli.command()
def report() -> None:
    """Generate a markdown report."""
    md = _tracker.export_markdown()
    click.echo(md)


@cli.command()
@click.argument("query")
@click.option("--limit", default=1, help="Number of papers to process")
def pipeline(query: str, limit: int) -> None:
    """Run end-to-end: fetch paper -> extract -> codegen -> track."""
    from paper_benchmark.pipeline import Pipeline

    pipe = Pipeline()
    base_code = "# placeholder base code\nimport torch\n"
    results = pipe.run(query, base_code, limit=limit)
    for pr in results:
        click.echo(f"Paper: {pr.paper.title}")
        click.echo(f"  Techniques: {len(pr.techniques)}")
        click.echo(f"  Modifications: {len(pr.modifications)}")
        click.echo(f"  Results logged: {pr.results_logged}")


@cli.command()
@click.argument("keywords")
@click.option("--limit", default=10, help="Number of papers to discover")
@click.option("--year-from", default=None, type=int, help="Earliest publication year")
def discover(keywords: str, limit: int, year_from: int | None) -> None:
    """Discover and rank recent papers by citation growth."""
    from paper_benchmark.discover import PaperDiscovery

    discovery = PaperDiscovery()
    ranked = discovery.discover(keywords, limit=limit, year_from=year_from)
    if not ranked:
        click.echo("No papers discovered.")
        return
    for i, rp in enumerate(ranked, 1):
        click.echo(
            f"{i}. [{rp.score:.0f}] {rp.paper.title} "
            f"(citations={rp.citation_count}, year={rp.year})"
        )


if __name__ == "__main__":
    cli()
