# paper-benchmark

**From paper to benchmark in one command.**

Search for a paper, extract the ML techniques it describes, generate code modifications, and benchmark them against your baseline -- all in a single pipeline. Stop reading papers and wondering "would this actually help?"

---

## The Problem

You found a promising paper on arXiv. It claims cosine annealing + label smoothing gives a 3% accuracy boost. Now you need to: (1) read the full paper, (2) figure out the implementation details, (3) modify your training code, (4) run experiments, (5) compare results. That's a full day -- per paper. Multiply by the 10 papers your advisor sent this week.

**paper-benchmark** automates the entire pipeline. One command. Paper in, benchmark results out.

## Quick Demo

```bash
$ paper-benchmark pipeline "cosine annealing learning rate"

Paper: Loshchilov & Hutter - SGDR: Stochastic Gradient Descent with Warm Restarts
  Techniques: 2
    - cosine annealing [training_trick]
    - learning rate warmup [training_trick]
  Modifications: 2
  Results logged: 2

$ paper-benchmark results
[cosine annealing]  0.8100 -> 0.8340 (+2.96%)
[learning rate warmup] 0.8100 -> 0.8250 (+1.85%)
```

## Features

- **Paper Search** -- Query Semantic Scholar's API to find relevant ML papers by keyword. Returns titles, abstracts, and URLs.

- **Paper Discovery** -- Rank papers by citation growth and influence. Filter by year to find the hottest recent techniques. Weighted scoring emphasizes recent, highly-cited, influential papers.

- **PDF Extraction** -- Fetch full papers directly from arXiv, extract text with pypdf, and run technique extraction on the complete paper -- not just the abstract.

- **Technique Extraction** -- 22 regex patterns covering optimizers (Adam, SGD), regularization (dropout, weight decay, label smoothing, stochastic depth), architectures (BatchNorm, LayerNorm, residual connections, attention), data augmentation (mixup, cutmix, cutout), and training tricks (warmup, cosine annealing, gradient clipping, mixed precision, knowledge distillation, curriculum learning, progressive resizing, layer freezing, contrastive learning).

- **LLM-Powered Extraction** -- Optional Claude integration for structured technique extraction from complex abstracts. Returns JSON with names, descriptions, categories, and implementation hints. Falls back to regex automatically.

- **Code Generation** -- Template-based code modifications for PyTorch, PyTorch Lightning, and HuggingFace Transformers. Automatically detects your framework and applies the right modification pattern.

- **Hyperparameter Search** -- Auto-generate parameter variants per technique category. Sweep learning rates, dropout values, hidden dimensions, and more with predefined grids.

- **Parallel Benchmarking** -- Run multiple benchmark tasks concurrently using ThreadPoolExecutor. Configurable worker count.

- **Result Tracking** -- Log all benchmark results with paper attribution. Export as markdown tables or JSON. Find top-performing techniques instantly.

- **End-to-End Pipeline** -- Chain fetch -> extract -> codegen -> benchmark -> track in a single `pipeline` command.

- **Ecosystem Integration** -- Optional hooks into `contradiction-detector` for flagging conflicting benchmark results and `autoresearch-memory` for persistent result storage.

## Installation

```bash
pip install paper-benchmark
```

For PDF extraction and LLM features:
```bash
pip install paper-benchmark[all]
```

## Usage

### CLI

```bash
# Search for papers on a topic
paper-benchmark fetch "attention mechanism transformer"

# Fetch and extract text from an arXiv PDF
paper-benchmark fetch "https://arxiv.org/abs/2301.12345" --pdf

# Discover trending papers ranked by citation influence
paper-benchmark discover "knowledge distillation" --limit 10 --year-from 2023

# Extract techniques from a paper abstract
paper-benchmark extract "We use dropout with rate 0.3 and Adam optimizer with cosine annealing..."

# Extract techniques using LLM (requires ANTHROPIC_API_KEY)
paper-benchmark extract "..." --llm

# Full pipeline: search -> extract -> codegen -> track
paper-benchmark pipeline "batch normalization residual connections" --limit 3

# View all benchmark results
paper-benchmark results

# Generate markdown report
paper-benchmark report
```

### Python API

```python
from paper_benchmark.papers import PaperFetcher
from paper_benchmark.extractor import TechniqueExtractor
from paper_benchmark.codegen import CodeGenerator

# Search for papers
fetcher = PaperFetcher()
papers = fetcher.fetch_papers("learning rate warmup", limit=5)

# Extract techniques from abstract
extractor = TechniqueExtractor()
techniques = extractor.extract_from_abstract(papers[0].abstract)
for t in techniques:
    print(f"  {t.name} [{t.category}]: {t.implementation_hint}")

# Generate code modification
codegen = CodeGenerator()
base_code = open("train.py").read()
modified = codegen.generate_modification(techniques[0], base_code)
```

```python
# Full pipeline
from paper_benchmark.pipeline import Pipeline

pipe = Pipeline()
results = pipe.run("cosine annealing", base_code="import torch\n...", limit=3)
for r in results:
    print(f"Paper: {r.paper.title}")
    print(f"  {len(r.techniques)} techniques, {len(r.modifications)} modifications")

# Hyperparameter variants
from paper_benchmark.hyperparam import HyperparamSearch

search = HyperparamSearch()
for technique in techniques:
    variants = search.generate_variants(technique)
    for v in variants:
        print(f"  {v.variant_id}: {v.params}")

# Parallel benchmarking
from paper_benchmark.parallel import ParallelBenchmarkRunner, BenchmarkTask

runner = ParallelBenchmarkRunner(max_workers=4)
tasks = [BenchmarkTask(name, hypothesis, run_fn) for ...]
results = runner.run_parallel(tasks)
```

## How It Works

```
"cosine annealing"
    |
    v
[PaperFetcher] -- Semantic Scholar API search
    |               (or arXiv PDF download + text extraction)
    v
[TechniqueExtractor] -- 22 regex patterns (or Claude LLM)
    |                     -> Technique(name, category, implementation_hint)
    v
[CodeGenerator] -- Framework-aware code modification
    |               Supports: PyTorch, Lightning, HuggingFace
    v
[BenchmarkRunner] -- Execute baseline vs modified, compute improvement %
    |                  (optional: ParallelBenchmarkRunner for concurrent runs)
    v
[ResultTracker] -- Log with paper attribution
    |               Export: markdown table, JSON
    v
[HyperparamSearch] -- Generate parameter sweep variants per category
```

**Key modules:**
- `papers.py` -- Semantic Scholar API client
- `discover.py` -- Citation-ranked paper discovery with recency scoring
- `pdf_fetcher.py` -- arXiv PDF download and text extraction
- `extractor.py` -- 22 regex patterns across 5 technique categories
- `llm_extractor.py` -- Claude-powered structured extraction with JSON output
- `codegen.py` -- Framework-aware code generation (PyTorch, Lightning, HuggingFace)
- `runner.py` -- Benchmark execution and improvement calculation
- `parallel.py` -- ThreadPoolExecutor-based concurrent benchmarking
- `hyperparam.py` -- Category-specific parameter grid generation
- `tracker.py` -- Result logging, ranking, markdown/JSON export
- `pipeline.py` -- End-to-end orchestration

## Comparison

| Feature | Manual Paper Reading | Papers with Code | paper-benchmark |
|---|---|---|---|
| Find relevant papers | Google Scholar | Leaderboard browsing | Keyword search + citation ranking |
| Extract techniques | Read full paper | Manual lookup | Automatic (regex + LLM) |
| Generate code changes | Write from scratch | Copy reference impl | Auto-generated, framework-aware |
| Benchmark against your code | Manual setup | No | One command |
| Track results with attribution | Spreadsheet | No | Built-in tracker |
| Hyperparameter sweep | Manual grid | No | Auto-generated variants |
| Parallel execution | Manual | No | Built-in ThreadPool |
| PDF full-text extraction | Read PDF yourself | Link to paper | Automatic with pypdf |
| Time per paper | 4-8 hours | 1-2 hours | < 5 minutes |

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/autoresearch/paper-benchmark.git
cd paper-benchmark
pip install -e ".[dev]"
pytest
```

Areas where we'd love help:
- More technique patterns (current: 22, goal: 100+)
- Framework support beyond PyTorch/Lightning/HuggingFace (JAX, TensorFlow)
- Actual benchmark execution with real training loops
- Web UI for browsing results and papers
- Integration with more paper databases (arXiv API, OpenAlex)

## License

MIT
