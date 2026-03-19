# paper-benchmark

Read a paper, extract ML techniques, implement modifications, and benchmark against baseline.

## Features

- Search Semantic Scholar for ML papers by keyword
- Extract techniques from paper abstracts using pattern matching
- Generate code modifications based on technique templates
- Benchmark modifications against baseline code
- Track results with paper attribution

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Search for a paper
paper-benchmark fetch "learning rate warmup"

# Extract techniques from a paper abstract
paper-benchmark extract "We propose a cosine annealing schedule..."

# Run a benchmark
paper-benchmark bench --technique "cosine_annealing" --base train.py

# View tracked results
paper-benchmark results

# Generate a report
paper-benchmark report
```

### Python API

```python
from paper_benchmark.extractor import TechniqueExtractor
from paper_benchmark.codegen import CodeGenerator

techniques = TechniqueExtractor.extract_from_abstract(abstract_text)
modified_code = CodeGenerator.generate_modification(techniques[0], base_code)
```

## Architecture

- `PaperFetcher` - Fetches papers from Semantic Scholar API
- `TechniqueExtractor` - Extracts ML techniques from abstracts
- `CodeGenerator` - Template-based code modifications
- `BenchmarkRunner` - Runs and compares experiments
- `ResultTracker` - Tracks and exports results
