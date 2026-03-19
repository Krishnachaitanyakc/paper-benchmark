"""CLI for autoresearch-bench."""

from __future__ import annotations

import click

from autoresearch_bench.extractor import TechniqueExtractor
from autoresearch_bench.papers import PaperFetcher
from autoresearch_bench.tracker import ResultTracker

_tracker = ResultTracker()
_extractor = TechniqueExtractor()
_fetcher = PaperFetcher()


@click.group()
def cli() -> None:
    """autoresearch-bench: Paper-to-benchmark pipeline."""


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of papers to fetch")
def fetch(query: str, limit: int) -> None:
    """Search Semantic Scholar for papers."""
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
def extract(text: str) -> None:
    """Extract ML techniques from text."""
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


if __name__ == "__main__":
    cli()
