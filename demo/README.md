# paper-benchmark Demo

## Quick Start

```bash
# From the project root
chmod +x demo/run_demo.sh
./demo/run_demo.sh
```

## What the Demo Shows

1. **Technique Extraction** -- Feeds four sample paper abstracts through the 22-pattern regex extractor and prints the detected techniques with categories and implementation hints.

2. **Code Generation** -- Takes `sample_baseline.py` (a minimal training script) and applies a dropout regularization modification using the template-based code generator.

3. **Hyperparameter Variants** -- Generates parameter sweep variants for a dropout technique, showing the automatic grid search configuration.

4. **Result Tracking** -- Simulates benchmark results from four well-known papers, logs them to the tracker, and exports a ranked markdown report.

## Files

- `sample_baseline.py` -- A simple simulated training loop that serves as the "before" code for code generation demos.
- `run_demo.sh` -- Self-contained shell script that runs all four demo steps. No network access or GPU required.

## Requirements

- Python 3.9+
- No external API keys needed (the demo uses only local components)
- Run from the project root so `PYTHONPATH=src` resolves correctly
