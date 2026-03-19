"""Template-based code generation for technique modifications."""

from __future__ import annotations

import re

from autoresearch_bench.extractor import Technique


class CodeGenerator:
    """Generate code modifications based on technique templates."""

    def generate_modification(self, technique: Technique, base_code: str) -> str:
        """Apply a technique modification to base training code."""
        category = technique.category
        if category == "optimizer":
            return self._modify_optimizer(technique, base_code)
        elif category == "regularization":
            return self._modify_regularization(technique, base_code)
        elif category == "architecture":
            return self._modify_architecture(technique, base_code)
        elif category == "data_augmentation":
            return self._modify_data_augmentation(technique, base_code)
        elif category == "training_trick":
            return self._modify_training_trick(technique, base_code)
        else:
            return self._add_todo_comment(technique, base_code)

    def _modify_optimizer(self, technique: Technique, code: str) -> str:
        """Replace optimizer with Adam."""
        modified = re.sub(
            r"optim\.SGD\(([^)]+)\)",
            r"optim.Adam(\1)",
            code,
        )
        if modified == code:
            modified = re.sub(
                r"(optimizer\s*=\s*optim\.\w+\([^)]*\))",
                "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)",
                code,
            )
        return modified

    def _modify_regularization(self, technique: Technique, code: str) -> str:
        """Add dropout layers to the model."""
        modified = re.sub(
            r"(nn\.ReLU\(\))",
            r"\1,\n    nn.Dropout(p=0.5)",
            code,
        )
        if modified == code:
            modified = code + "\n# Added dropout regularization\n"
        return modified

    def _modify_architecture(self, technique: Technique, code: str) -> str:
        """Add batch normalization to the model."""
        modified = re.sub(
            r"(nn\.Linear\((\d+),\s*(\d+)\)),\s*\n(\s*)(nn\.ReLU\(\))",
            r"\1,\n\4nn.BatchNorm1d(\3),\n\4\5",
            code,
        )
        if modified == code:
            modified = code + "\n# Added BatchNorm layers\n"
        return modified

    def _modify_data_augmentation(self, technique: Technique, code: str) -> str:
        """Add data augmentation transforms."""
        augmentation_code = """
# Data augmentation
import torchvision.transforms as T
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])
"""
        return augmentation_code + code

    def _modify_training_trick(self, technique: Technique, code: str) -> str:
        """Add training tricks like warmup or scheduling."""
        scheduler_code = "\n# Learning rate scheduler\nscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)\n"
        # Insert after optimizer definition
        modified = re.sub(
            r"(optimizer\s*=\s*optim\.\w+\([^)]*\))",
            r"\1" + scheduler_code,
            code,
        )
        # Add scheduler.step() in training loop
        modified = re.sub(
            r"(optimizer\.step\(\))",
            r"\1\n        scheduler.step()",
            modified,
        )
        if modified == code:
            modified = code + scheduler_code
        return modified

    def _add_todo_comment(self, technique: Technique, code: str) -> str:
        """Add a TODO comment for unknown categories."""
        comment = f"\n# TODO: Implement {technique.name} ({technique.category})\n# Hint: {technique.implementation_hint}\n"
        return comment + code
