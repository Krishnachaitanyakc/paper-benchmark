"""Tests for hyperparameter search."""

from paper_benchmark.extractor import Technique
from paper_benchmark.hyperparam import HyperparamSearch, HyperparamVariant


class TestHyperparamSearch:
    def setup_method(self):
        self.search = HyperparamSearch()

    def test_generate_variants_regularization(self):
        technique = Technique("dropout", "desc", "regularization", "hint")
        variants = self.search.generate_variants(technique)
        assert len(variants) == 3  # dropout_p has 3 values
        assert all(isinstance(v, HyperparamVariant) for v in variants)
        assert variants[0].variant_id == "dropout_v0"

    def test_generate_variants_optimizer(self):
        technique = Technique("adam", "desc", "optimizer", "hint")
        variants = self.search.generate_variants(technique)
        assert len(variants) == 3  # lr has 3 values

    def test_generate_variants_unknown_category(self):
        technique = Technique("unknown", "desc", "unknown_cat", "hint")
        variants = self.search.generate_variants(technique)
        assert len(variants) == 1
        assert variants[0].variant_id == "unknown_default"
        assert variants[0].params == {}

    def test_variant_params_differ(self):
        technique = Technique("dropout", "desc", "regularization", "hint")
        variants = self.search.generate_variants(technique)
        primary_values = [v.params["dropout_p"] for v in variants]
        assert len(set(primary_values)) == 3  # all different

    def test_custom_param_grid(self):
        custom_grid = {"optimizer": {"lr": [0.001, 0.01]}}
        search = HyperparamSearch(param_grids=custom_grid)
        technique = Technique("sgd", "desc", "optimizer", "hint")
        variants = search.generate_variants(technique)
        assert len(variants) == 2

    def test_training_trick_variants(self):
        technique = Technique("warmup", "desc", "training_trick", "hint")
        variants = self.search.generate_variants(technique)
        assert len(variants) == 3  # warmup_steps has 3 values
