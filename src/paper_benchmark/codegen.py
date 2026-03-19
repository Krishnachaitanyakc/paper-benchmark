"""Template-based code generation for technique modifications."""

from __future__ import annotations

import re

from paper_benchmark.extractor import Technique


class CodeGenerator:
    """Generate code modifications based on technique templates."""

    def generate_modification(self, technique: Technique, base_code: str) -> str:
        """Apply a technique modification to base training code."""
        # Detect framework and dispatch accordingly
        if self._is_lightning(base_code):
            return self._modify_lightning(technique, base_code)
        if self._is_huggingface(base_code):
            return self._modify_huggingface(technique, base_code)

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

    @staticmethod
    def _is_lightning(code: str) -> bool:
        return "pytorch_lightning" in code or "lightning.pytorch" in code or "pl.LightningModule" in code

    @staticmethod
    def _is_huggingface(code: str) -> bool:
        return "transformers" in code and ("Trainer" in code or "AutoModel" in code)

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

    def _modify_lightning(self, technique: Technique, code: str) -> str:
        """Modify PyTorch Lightning code to apply technique."""
        category = technique.category
        if category == "optimizer":
            modified = re.sub(
                r"(def configure_optimizers\(self\)[^:]*:.*?\n\s+)(return\s+.*)",
                r"\1# Modified: using Adam optimizer\n        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)",
                code,
                flags=re.DOTALL,
            )
            if modified != code:
                return modified
            # If no configure_optimizers found, add one
            modified = re.sub(
                r"(class\s+\w+\(.*?LightningModule.*?\):.*?\n)",
                r'\1\n    def configure_optimizers(self):\n        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)\n',
                code,
            )
            if modified != code:
                return modified
        elif category == "regularization":
            modified = re.sub(
                r"(nn\.ReLU\(\))",
                r"\1,\n            nn.Dropout(p=0.5)",
                code,
            )
            if modified != code:
                return modified
        elif category == "training_trick":
            # Add LR scheduler to configure_optimizers
            modified = re.sub(
                r"(def configure_optimizers\(self\)[^:]*:.*?\n\s+)(return\s+(torch\.optim\.\w+|optim\.\w+)\(([^)]*)\))",
                r'\1optimizer = \3(\4)\n        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)\n        return [optimizer], [scheduler]',
                code,
                flags=re.DOTALL,
            )
            if modified != code:
                return modified
        return self._add_todo_comment(technique, code)

    def _modify_huggingface(self, technique: Technique, code: str) -> str:
        """Modify HuggingFace Transformers code to apply technique."""
        category = technique.category
        if category == "optimizer":
            modified = re.sub(
                r"(TrainingArguments\()",
                r'\1\n        learning_rate=2e-5,\n        weight_decay=0.01,',
                code,
            )
            if modified != code:
                return modified
        elif category == "regularization":
            modified = re.sub(
                r"(TrainingArguments\()",
                r'\1\n        label_smoothing_factor=0.1,',
                code,
            )
            if modified != code:
                return modified
        elif category == "training_trick":
            modified = re.sub(
                r"(TrainingArguments\()",
                r'\1\n        warmup_steps=500,\n        lr_scheduler_type="cosine",',
                code,
            )
            if modified != code:
                return modified
        elif category == "data_augmentation":
            augmentation_import = "# Data augmentation for HuggingFace\nfrom datasets import Dataset\n"
            return augmentation_import + code
        return self._add_todo_comment(technique, code)

    def _add_todo_comment(self, technique: Technique, code: str) -> str:
        """Add a TODO comment for unknown categories."""
        comment = f"\n# TODO: Implement {technique.name} ({technique.category})\n# Hint: {technique.implementation_hint}\n"
        return comment + code
