"""Tests for technique extractor."""

import pytest
from autoresearch_bench.extractor import TechniqueExtractor, Technique


class TestTechnique:
    def test_technique_creation(self):
        t = Technique(
            name="dropout",
            description="Randomly drop units during training",
            category="regularization",
            implementation_hint="Add nn.Dropout(p=0.5) after linear layers",
        )
        assert t.name == "dropout"
        assert t.category == "regularization"


class TestTechniqueExtractor:
    def setup_method(self):
        self.extractor = TechniqueExtractor()

    def test_extract_optimizer_technique(self):
        abstract = "We propose using Adam optimizer with weight decay for improved convergence."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 1
        categories = [t.category for t in techniques]
        assert "optimizer" in categories

    def test_extract_regularization_technique(self):
        abstract = "We introduce dropout regularization to prevent overfitting in deep networks."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 1
        categories = [t.category for t in techniques]
        assert "regularization" in categories

    def test_extract_architecture_technique(self):
        abstract = "We propose a residual connection architecture with skip connections for deeper networks."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 1
        categories = [t.category for t in techniques]
        assert "architecture" in categories

    def test_extract_data_augmentation_technique(self):
        abstract = "We apply data augmentation including random crop and mixup to improve generalization."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 1
        categories = [t.category for t in techniques]
        assert "data_augmentation" in categories

    def test_extract_training_trick(self):
        abstract = "We use learning rate warmup and cosine annealing schedule for stable training."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 1
        categories = [t.category for t in techniques]
        assert "training_trick" in categories

    def test_extract_multiple_techniques(self):
        abstract = (
            "We combine dropout regularization with Adam optimizer "
            "and learning rate warmup for best results."
        )
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 2

    def test_extract_no_techniques(self):
        abstract = "This paper discusses the history of computing."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert techniques == []

    def test_technique_has_implementation_hint(self):
        abstract = "We use batch normalization after each convolutional layer."
        techniques = self.extractor.extract_from_abstract(abstract)
        assert len(techniques) >= 1
        for t in techniques:
            assert t.implementation_hint != ""
