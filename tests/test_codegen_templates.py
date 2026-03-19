"""Tests for Lightning and HuggingFace codegen templates."""

from paper_benchmark.codegen import CodeGenerator
from paper_benchmark.extractor import Technique

LIGHTNING_CODE = """
import pytorch_lightning as pl
import torch
import torch.nn as nn

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
"""

HUGGINGFACE_CODE = """
from transformers import AutoModel, TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
)

trainer = Trainer(model=model, args=training_args)
trainer.train()
"""


class TestLightningTemplate:
    def setup_method(self):
        self.gen = CodeGenerator()

    def test_is_lightning_detection(self):
        assert CodeGenerator._is_lightning(LIGHTNING_CODE)
        assert not CodeGenerator._is_lightning("import torch")

    def test_lightning_regularization(self):
        technique = Technique("dropout", "add dropout", "regularization", "hint")
        modified = self.gen.generate_modification(technique, LIGHTNING_CODE)
        assert "Dropout" in modified

    def test_lightning_unknown_category(self):
        technique = Technique("unknown", "desc", "unknown_cat", "hint")
        modified = self.gen.generate_modification(technique, LIGHTNING_CODE)
        assert "TODO" in modified


class TestHuggingFaceTemplate:
    def setup_method(self):
        self.gen = CodeGenerator()

    def test_is_huggingface_detection(self):
        assert CodeGenerator._is_huggingface(HUGGINGFACE_CODE)
        assert not CodeGenerator._is_huggingface("import torch")

    def test_huggingface_optimizer(self):
        technique = Technique("adam", "use adam", "optimizer", "hint")
        modified = self.gen.generate_modification(technique, HUGGINGFACE_CODE)
        assert "learning_rate" in modified or "weight_decay" in modified

    def test_huggingface_regularization(self):
        technique = Technique("label smoothing", "smooth", "regularization", "hint")
        modified = self.gen.generate_modification(technique, HUGGINGFACE_CODE)
        assert "label_smoothing" in modified

    def test_huggingface_training_trick(self):
        technique = Technique("warmup", "warmup lr", "training_trick", "hint")
        modified = self.gen.generate_modification(technique, HUGGINGFACE_CODE)
        assert "warmup_steps" in modified

    def test_huggingface_data_augmentation(self):
        technique = Technique("augment", "data aug", "data_augmentation", "hint")
        modified = self.gen.generate_modification(technique, HUGGINGFACE_CODE)
        assert modified != HUGGINGFACE_CODE
