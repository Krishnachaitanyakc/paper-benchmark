"""Extract ML techniques from paper abstracts using pattern matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

TECHNIQUE_PATTERNS: List[Tuple[str, str, str, str]] = [
    # (regex_pattern, name, category, implementation_hint)
    (
        r"(?:adam|adamw|adagrad|rmsprop)\b.*?(?:optim|weight.?decay|converg)",
        "Adam optimizer",
        "optimizer",
        "Replace optimizer with torch.optim.Adam(params, lr=lr, weight_decay=1e-4)",
    ),
    (
        r"(?:sgd|momentum|nesterov)\b.*?(?:optim|converg|momentum)",
        "SGD with momentum",
        "optimizer",
        "Use torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)",
    ),
    (
        r"\b(?:dropout)\b.*?(?:regulariz|overfit|prevent|rate)",
        "dropout",
        "regularization",
        "Add nn.Dropout(p=0.5) after activation layers",
    ),
    (
        r"\b(?:weight.?decay|l2.?regulariz)\b",
        "weight decay",
        "regularization",
        "Add weight_decay=1e-4 to optimizer",
    ),
    (
        r"\b(?:label.?smooth)\b",
        "label smoothing",
        "regularization",
        "Use label smoothing in cross-entropy loss with smoothing=0.1",
    ),
    (
        r"\b(?:batch.?norm\w*|batchnorm)\b",
        "batch normalization",
        "architecture",
        "Add nn.BatchNorm1d/2d after linear/conv layers before activation",
    ),
    (
        r"\b(?:layer.?norm\w*|layernorm)\b",
        "layer normalization",
        "architecture",
        "Add nn.LayerNorm after sublayers",
    ),
    (
        r"\b(?:resid\w*\s+connect\w*|skip\s+connect\w*|shortcut\w*)\b.*?(?:connect|layer|block|deep|network)",
        "residual connections",
        "architecture",
        "Add skip connections: output = layer(x) + x",
    ),
    (
        r"\b(?:attention|self.?attention|transformer)\b.*?(?:mechanism|layer|head)",
        "attention mechanism",
        "architecture",
        "Add nn.MultiheadAttention layer",
    ),
    (
        r"\b(?:data.?augment|random.?crop|random.?flip|mixup|cutout|cutmix)\b",
        "data augmentation",
        "data_augmentation",
        "Add torchvision.transforms with RandomCrop, RandomHorizontalFlip",
    ),
    (
        r"\b(?:learning.?rate.?warm|warmup|warm.?up)\b.*?(?:schedul|train|first|epoch)",
        "learning rate warmup",
        "training_trick",
        "Linear warmup: lr = base_lr * min(1, step / warmup_steps)",
    ),
    (
        r"\b(?:cosine.?anneal|cosine.?schedul|cosine.?decay)\b",
        "cosine annealing",
        "training_trick",
        "Use torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)",
    ),
    (
        r"\b(?:gradient.?clip|clip.?grad|grad.?clip)\b",
        "gradient clipping",
        "training_trick",
        "Add torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
    ),
    (
        r"\b(?:mixed.?precision|fp16|half.?precision|amp)\b",
        "mixed precision training",
        "training_trick",
        "Use torch.cuda.amp.autocast() and GradScaler",
    ),
    (
        r"\b(?:contrastive.?learn\w*|simclr|moco|byol|contrastive.?loss)\b",
        "contrastive learning",
        "training_trick",
        "Implement contrastive loss: NT-Xent or InfoNCE with positive/negative pairs",
    ),
    (
        r"\b(?:mixup)\b.*?(?:train|augment|interpolat|regulariz|sample)",
        "mixup",
        "data_augmentation",
        "Interpolate pairs: x_mix = lambda*x_i + (1-lambda)*x_j, lambda ~ Beta(alpha, alpha)",
    ),
    (
        r"\b(?:cutmix)\b",
        "cutmix",
        "data_augmentation",
        "Replace image region with patch from another image, adjust labels proportionally",
    ),
    (
        r"\b(?:knowledge.?distill\w*|teacher.?student|soft.?label|dark.?knowledge)\b",
        "knowledge distillation",
        "training_trick",
        "Train student with KL-div loss between student and teacher logits (temperature=4)",
    ),
    (
        r"\b(?:progressive.?resiz\w*|curriculum.?resiz\w*|multi.?scale.?train)\b",
        "progressive resizing",
        "training_trick",
        "Start training with small images and progressively increase resolution each phase",
    ),
    (
        r"\b(?:stochastic.?depth|drop.?path|survival.?prob)\b",
        "stochastic depth",
        "regularization",
        "Randomly drop entire residual blocks during training with linearly increasing drop rate",
    ),
    (
        r"\b(?:layer.?freez\w*|freeze.?layer|frozen.?layer|gradual.?unfreez)\b",
        "layer freezing",
        "training_trick",
        "Freeze pretrained layers initially, then gradually unfreeze for fine-tuning",
    ),
    (
        r"\b(?:curriculum.?learn\w*|self.?paced|easy.?to.?hard)\b",
        "curriculum learning",
        "training_trick",
        "Order training samples from easy to hard using a difficulty scoring function",
    ),
]


@dataclass
class Technique:
    """An ML technique extracted from a paper."""

    name: str
    description: str
    category: str
    implementation_hint: str


class TechniqueExtractor:
    """Extract ML techniques from paper abstracts."""

    def extract_from_abstract(self, text: str) -> List[Technique]:
        """Extract techniques from abstract text using pattern matching."""
        text_lower = text.lower()
        found: List[Technique] = []
        seen_names: set = set()

        for pattern, name, category, hint in TECHNIQUE_PATTERNS:
            if re.search(pattern, text_lower) and name not in seen_names:
                seen_names.add(name)
                # Extract a sentence containing the match for description
                match = re.search(pattern, text_lower)
                if match:
                    start = max(0, text_lower.rfind(".", 0, match.start()) + 1)
                    end = text_lower.find(".", match.end())
                    if end == -1:
                        end = len(text_lower)
                    description = text[start:end].strip()
                else:
                    description = name

                found.append(
                    Technique(
                        name=name,
                        description=description,
                        category=category,
                        implementation_hint=hint,
                    )
                )

        return found
