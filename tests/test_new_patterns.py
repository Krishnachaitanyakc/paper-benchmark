"""Tests for new extraction patterns."""

from paper_benchmark.extractor import TechniqueExtractor


class TestNewPatterns:
    def setup_method(self):
        self.extractor = TechniqueExtractor()

    def test_contrastive_learning(self):
        text = "We use contrastive learning with SimCLR to learn visual representations."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "contrastive learning" in names

    def test_mixup(self):
        text = "Mixup training augments data by interpolating between training samples."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "mixup" in names

    def test_cutmix(self):
        text = "CutMix replaces a region of an image with a patch from another image."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "cutmix" in names

    def test_knowledge_distillation(self):
        text = "Knowledge distillation transfers dark knowledge from a teacher model to a student."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "knowledge distillation" in names

    def test_progressive_resizing(self):
        text = "Progressive resizing starts training with small images and increases resolution."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "progressive resizing" in names

    def test_stochastic_depth(self):
        text = "Stochastic depth randomly drops entire residual blocks during training."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "stochastic depth" in names

    def test_layer_freezing(self):
        text = "Layer freezing keeps pretrained weights fixed during initial fine-tuning."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "layer freezing" in names

    def test_curriculum_learning(self):
        text = "Curriculum learning orders samples from easy to hard during training."
        techniques = self.extractor.extract_from_abstract(text)
        names = [t.name for t in techniques]
        assert "curriculum learning" in names
