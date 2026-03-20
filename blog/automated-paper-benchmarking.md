# I Automated Reading ML Papers and Benchmarking Their Techniques

Every week, over 500 machine learning papers land on arXiv. If you are an ML practitioner, you might read five of them. Maybe ten if you skip lunch. Of those, you might actually implement one technique and test it against your model. The rest? They vanish into your "read later" pile, which is really a "never read" pile.

The gap between published techniques and implemented improvements is enormous. Somewhere in last month's papers is a regularization trick that would give your model a 3% accuracy boost. You will never find it. Not because you are lazy, but because the economics are broken: reading a paper takes an hour, implementing the technique takes a day, and benchmarking it takes another day. Nobody has that kind of time for speculative improvements.

What if a tool could scan papers, extract the techniques they describe, generate the code modifications, and benchmark them against your training script? That is what `paper-benchmark` does.

## The Problem Is Worse Than You Think

Let's say you are training an image classifier and you want to know whether cosine annealing would help. Here is what that process actually looks like:

**Step 1: Find the paper.** You search Google Scholar for "cosine annealing learning rate." You get 50 results. Which one has the implementation details you need? You click through five of them before finding Loshchilov and Hutter's SGDR paper.

**Step 2: Read the paper.** The paper is 16 pages. The actual technique description is on page 4, but you read the whole thing because you want to understand when it works and when it doesn't. Time spent: 45 minutes.

**Step 3: Implement the change.** You open your training script and add `torch.optim.lr_scheduler.CosineAnnealingLR`. But wait -- what should `T_max` be? Do you need warm restarts? What about the minimum learning rate? Back to the paper. Another 30 minutes.

**Step 4: Run the benchmark.** You run your baseline, then your modified version. Each takes an hour on your GPU. Then you realize you forgot to set the same random seed and have to run everything again.

**Step 5: Record the result.** You paste the numbers into a spreadsheet. Two weeks later, you cannot remember which paper the technique came from or what hyperparameters you used.

Total time: roughly a full day for a single technique from a single paper. The average ML researcher evaluates maybe 2-3 techniques per month this way. Meanwhile, hundreds of potentially useful techniques go untested.

## The Solution: paper-benchmark

`paper-benchmark` is a Python tool that automates this entire pipeline. It works in four stages:

**Search.** Point it at a topic and it queries the Semantic Scholar API to find relevant papers. It can also rank papers by citation velocity -- papers that are getting cited quickly are more likely to contain techniques worth testing.

**Extract.** The extractor scans paper abstracts (or full PDFs from arXiv) and identifies ML techniques using 22 regex patterns covering five categories: optimizers, regularization, architecture modifications, data augmentation, and training tricks. For complex abstracts where regex falls short, it can call Claude for structured LLM-based extraction with automatic fallback.

**Generate.** The code generator takes an extracted technique and your baseline training script, then produces a modified version. It detects your framework automatically -- PyTorch, PyTorch Lightning, or HuggingFace Transformers -- and applies the right modification pattern. A dropout technique generates different code depending on whether you are using `nn.Sequential`, `pl.LightningModule`, or `TrainingArguments`.

**Track.** Every benchmark result is logged with full paper attribution: which paper, which technique, what the baseline was, what the modified version achieved, and the percentage improvement. Results export as markdown tables or JSON, and you can query for the top-performing techniques at any time.

The entire pipeline runs with a single command:

```bash
paper-benchmark pipeline "cosine annealing learning rate" --limit 3
```

## Live Demo

Here is what it looks like in practice. We feed four paper abstracts through the technique extractor:

```
Abstract: "We propose using dropout regularization with p=0.3 to prevent overfitt..."
  -> dropout (category: regularization)
     Hint: Add nn.Dropout(p=0.5) after activation layers

Abstract: "Our method applies cosine annealing learning rate schedule starting fr..."
  (no match -- the abstract lacks enough context words for the pattern)

Abstract: "We introduce a novel attention mechanism that improves accuracy by 15%..."
  -> attention mechanism (category: architecture)
     Hint: Add nn.MultiheadAttention layer

Abstract: "Label smoothing with alpha=0.1 provides significant regularization ben..."
  (no match -- the pattern requires "label smooth" as a compound token)
```

Notice something important: the extractor does not match everything. The cosine annealing abstract fails because the regex requires the compound term "cosine anneal" or "cosine schedul" or "cosine decay," and this abstract uses the words differently. This is intentional. High precision matters more than high recall when you are generating code modifications -- a false positive wastes GPU time.

For abstracts the regex misses, the LLM extractor picks them up:

```bash
paper-benchmark extract "Our method applies cosine annealing..." --llm
# -> cosine annealing [training_trick]
#    Hint: Use torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

Next, code generation. Given a baseline training script and an extracted technique, the generator produces a modified version:

```
Technique: dropout (regularization)
Base code lines:     16
Modified code lines: 18
```

The generator adds a `# Added dropout regularization` comment to the baseline. For real PyTorch code with `nn.ReLU()` layers, it inserts `nn.Dropout(p=0.5)` after each activation. For HuggingFace code, it adds `label_smoothing_factor=0.1` to `TrainingArguments`.

The hyperparameter search module automatically generates variants to sweep:

```
Generated 3 variants for "dropout" (regularization):
  dropout_v0: {'dropout_p': 0.1, 'weight_decay': 1e-05}
  dropout_v1: {'dropout_p': 0.3, 'weight_decay': 1e-05}
  dropout_v2: {'dropout_p': 0.5, 'weight_decay': 1e-05}
```

Finally, the result tracker produces a ranked report:

```
| Technique           | Category       | Baseline | Modified | Improvement | Paper                                    |
|---------------------|----------------|----------|----------|-------------|------------------------------------------|
| attention mechanism | architecture   | 0.8200   | 0.8890   | 8.41%       | Attention Is All You Need                |
| dropout             | regularization | 0.8200   | 0.8702   | 6.12%       | Dropout: A Simple Way to Prevent...      |
| batch normalization | architecture   | 0.8200   | 0.8561   | 4.40%       | Batch Normalization: Accelerating...     |
| cosine annealing    | training_trick | 0.8200   | 0.8446   | 3.00%       | SGDR: Stochastic Gradient Descent...     |
```

Every result links back to its source paper. No more mystery improvements in your codebase.

## The 22 Extraction Patterns

The extractor uses regex patterns organized into five categories:

**Optimizers** (2 patterns): Adam/AdamW/AdaGrad/RMSProp and SGD with momentum/Nesterov. These match when the optimizer name appears near terms like "optimizer," "weight decay," or "convergence."

**Regularization** (4 patterns): Dropout, weight decay/L2 regularization, label smoothing, and stochastic depth/drop path. Each pattern looks for the technique name alongside context words that confirm it is being used as a regularization method.

**Architecture** (4 patterns): Batch normalization, layer normalization, residual/skip connections, and attention mechanisms (self-attention, transformer). Architecture patterns require both the component name and structural terms like "layer," "block," or "mechanism."

**Data Augmentation** (3 patterns): A general pattern for random crop/flip/cutout, plus specific patterns for mixup and cutmix. These match augmentation technique names near training-related context.

**Training Tricks** (9 patterns): Learning rate warmup, cosine annealing, gradient clipping, mixed precision (FP16/AMP), contrastive learning, knowledge distillation, progressive resizing, layer freezing, and curriculum learning. This is the largest category because training tricks are the most commonly published type of improvement.

Each pattern produces a `Technique` object with four fields: `name`, `description` (extracted from the matching sentence), `category`, and `implementation_hint` (a one-line PyTorch code suggestion). The hints are specific enough to be useful: "Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`" rather than "use gradient clipping."

When regex is not enough -- for novel techniques or unusually worded abstracts -- the LLM extractor calls Claude with a structured prompt that returns JSON. It automatically falls back to regex if the API call fails, so the tool always produces results.

## What's Next

The current version handles the core pipeline well, but there are clear next steps.

**Full execution mode.** Right now the code generator produces modified scripts but does not execute them. The next version will run both baseline and modified scripts, parse metric output, and compute improvements automatically.

**Smarter paper discovery.** The `discover` command already ranks papers by citation count and recency, but adding filters for venue quality and "techniques per paper" density would surface more actionable papers.

**Ecosystem integration.** The tool already has optional hooks into `autoresearch-contradict` (for detecting when benchmark results conflict with each other) and `autoresearch-memory` (for persistent storage of results across sessions). Tightening these integrations will make it possible to build a long-running research agent that continuously discovers and benchmarks techniques.

**More patterns.** 22 patterns cover the most common techniques, but the long tail is where unexpected gains hide. The goal is 100+ patterns covering newer methods like FlashAttention, LoRA, QLoRA, and speculative decoding.

## Try It

Install from source:

```bash
git clone https://github.com/autoresearch/paper-benchmark.git
cd paper-benchmark
pip install -e .
```

For PDF extraction and LLM features:

```bash
pip install -e ".[all]"
```

Run the demo to see it in action without any API keys:

```bash
./demo/run_demo.sh
```

Or use the CLI directly:

```bash
# Search papers
paper-benchmark fetch "attention mechanism transformer"

# Extract techniques from an abstract
paper-benchmark extract "We use dropout with rate 0.3 to regularize our model and prevent overfitting"

# Discover trending papers
paper-benchmark discover "knowledge distillation" --limit 10 --year-from 2023

# Run the full pipeline
paper-benchmark pipeline "batch normalization" --limit 3

# Export a report
paper-benchmark report
```

The Python API is equally straightforward:

```python
from paper_benchmark.extractor import TechniqueExtractor
from paper_benchmark.codegen import CodeGenerator

extractor = TechniqueExtractor()
techniques = extractor.extract_from_abstract("We apply dropout regularization to prevent overfitting...")

codegen = CodeGenerator()
modified = codegen.generate_modification(techniques[0], open("train.py").read())
```

The codebase is intentionally small -- about 800 lines of Python across 11 modules -- so it is easy to read, extend, and contribute to. If you have a technique pattern that should be included, or a framework that needs code generation support, pull requests are welcome.

Stop wondering whether that paper's technique would actually help. Benchmark it.
