#!/bin/bash
set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Ensure httpx is available (papers.py requires it)
python3 -c "import httpx" 2>/dev/null || {
    echo "Installing httpx dependency..."
    python3 -m pip install --quiet httpx 2>/dev/null || pip3 install --quiet --target /tmp/pb_deps httpx 2>/dev/null || true
}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
# Include fallback deps location if needed
python3 -c "import httpx" 2>/dev/null || export PYTHONPATH="/tmp/pb_deps:$PYTHONPATH"

echo "============================================="
echo "    paper-benchmark  --  Interactive Demo"
echo "============================================="
echo ""

echo "--- Step 1: Extract techniques from paper abstracts ---"
echo ""
python3 -c "
from paper_benchmark.extractor import TechniqueExtractor
ext = TechniqueExtractor()

abstracts = [
    'We propose using dropout regularization with p=0.3 to prevent overfitting in deep networks.',
    'Our method applies cosine annealing learning rate schedule starting from 0.01.',
    'We introduce a novel attention mechanism that improves accuracy by 15% over the baseline transformer layer.',
    'Label smoothing with alpha=0.1 provides significant regularization benefits for classification.',
]

for abstract in abstracts:
    print(f'  Abstract: \"{abstract[:70]}...\"')
    techniques = ext.extract_from_abstract(abstract)
    for t in techniques:
        print(f'    -> {t.name} (category: {t.category})')
        print(f'       Hint: {t.implementation_hint}')
    print()
"

echo ""
echo "--- Step 2: Generate code modifications ---"
echo ""
python3 -c "
from paper_benchmark.codegen import CodeGenerator
from paper_benchmark.extractor import Technique

gen = CodeGenerator()
base_code = open('demo/sample_baseline.py').read()

tech = Technique(
    name='dropout',
    description='Add dropout regularization with p=0.3',
    category='regularization',
    implementation_hint='Add nn.Dropout(p=0.5) after activation layers',
)

modified = gen.generate_modification(tech, base_code)
print('  Technique: dropout (regularization)')
print('  Base code lines:    ', base_code.count(chr(10)))
print('  Modified code lines:', modified.count(chr(10)))
print()
print('  --- Modified code preview ---')
for line in modified.splitlines()[:20]:
    print(f'  | {line}')
print('  ...')
"

echo ""
echo "--- Step 3: Hyperparameter variant generation ---"
echo ""
python3 -c "
from paper_benchmark.hyperparam import HyperparamSearch
from paper_benchmark.extractor import Technique

search = HyperparamSearch()

tech = Technique(
    name='dropout',
    description='Dropout regularization',
    category='regularization',
    implementation_hint='Add dropout layers',
)

variants = search.generate_variants(tech)
print(f'  Generated {len(variants)} variants for \"{tech.name}\" ({tech.category}):')
for v in variants:
    print(f'    {v.variant_id}: {v.params}')
"

echo ""
echo "--- Step 4: Track and report benchmark results ---"
echo ""
python3 -c "
from paper_benchmark.tracker import ResultTracker, BenchmarkResult
from paper_benchmark.extractor import Technique
from paper_benchmark.papers import Paper

tracker = ResultTracker()

# Simulate several benchmark results
entries = [
    ('Dropout: A Simple Way to Prevent Overfitting',
     'https://arxiv.org/abs/1207.0580',
     'dropout', 'regularization', 0.8200, 0.8702, 'Dropout prevents overfitting'),
    ('Batch Normalization: Accelerating Deep Network Training',
     'https://arxiv.org/abs/1502.03167',
     'batch normalization', 'architecture', 0.8200, 0.8561, 'BatchNorm stabilizes training'),
    ('Attention Is All You Need',
     'https://arxiv.org/abs/1706.03762',
     'attention mechanism', 'architecture', 0.8200, 0.8890, 'Self-attention captures long-range dependencies'),
    ('SGDR: Stochastic Gradient Descent with Warm Restarts',
     'https://arxiv.org/abs/1608.03983',
     'cosine annealing', 'training_trick', 0.8200, 0.8446, 'Cosine LR schedule improves convergence'),
]

for title, url, tech_name, cat, base, mod, hyp in entries:
    paper = Paper(title=title, abstract='...', url=url)
    tech = Technique(name=tech_name, description=tech_name, category=cat, implementation_hint='')
    result = BenchmarkResult(
        baseline_metric=base,
        modified_metric=mod,
        improvement_pct=((mod - base) / base) * 100,
        hypothesis=hyp,
        technique_name=tech_name,
    )
    tracker.log_result(paper, tech, result)

print(tracker.export_markdown())

print()
best = tracker.find_best_techniques(top_n=3)
print('  Top 3 techniques:')
for r in best:
    print(f'    {r[\"technique_name\"]}: +{r[\"improvement_pct\"]:.2f}%')
"

echo ""
echo "============================================="
echo "  Demo complete. See README.md for more."
echo "============================================="
