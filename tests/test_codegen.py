"""Tests for code generator."""

import pytest
from paper_benchmark.codegen import CodeGenerator
from paper_benchmark.extractor import Technique


SAMPLE_TRAIN_PY = """
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
"""


class TestCodeGenerator:
    def setup_method(self):
        self.generator = CodeGenerator()

    def test_generate_optimizer_modification(self):
        technique = Technique(
            name="Adam optimizer",
            description="Use Adam instead of SGD",
            category="optimizer",
            implementation_hint="Replace SGD with Adam",
        )
        modified = self.generator.generate_modification(technique, SAMPLE_TRAIN_PY)
        assert "Adam" in modified
        assert modified != SAMPLE_TRAIN_PY

    def test_generate_regularization_modification(self):
        technique = Technique(
            name="dropout",
            description="Add dropout layers",
            category="regularization",
            implementation_hint="Add nn.Dropout(p=0.5)",
        )
        modified = self.generator.generate_modification(technique, SAMPLE_TRAIN_PY)
        assert "Dropout" in modified

    def test_generate_training_trick_modification(self):
        technique = Technique(
            name="learning rate warmup",
            description="Warmup the learning rate",
            category="training_trick",
            implementation_hint="Linear warmup for first 10 epochs",
        )
        modified = self.generator.generate_modification(technique, SAMPLE_TRAIN_PY)
        assert modified != SAMPLE_TRAIN_PY

    def test_generate_architecture_modification(self):
        technique = Technique(
            name="batch normalization",
            description="Add batch norm layers",
            category="architecture",
            implementation_hint="Add BatchNorm1d after linear layers",
        )
        modified = self.generator.generate_modification(technique, SAMPLE_TRAIN_PY)
        assert "BatchNorm" in modified

    def test_generate_data_augmentation_modification(self):
        technique = Technique(
            name="random noise",
            description="Add random noise to inputs",
            category="data_augmentation",
            implementation_hint="Add Gaussian noise",
        )
        modified = self.generator.generate_modification(technique, SAMPLE_TRAIN_PY)
        assert modified != SAMPLE_TRAIN_PY

    def test_unknown_category_returns_commented_code(self):
        technique = Technique(
            name="unknown",
            description="Unknown technique",
            category="unknown_category",
            implementation_hint="Do something",
        )
        modified = self.generator.generate_modification(technique, SAMPLE_TRAIN_PY)
        assert "# TODO" in modified or modified != SAMPLE_TRAIN_PY
