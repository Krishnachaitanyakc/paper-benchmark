"""Hyperparameter variant generation for extracted techniques."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from paper_benchmark.extractor import Technique


# Default parameter grids per category
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "regularization": {
        "dropout_p": [0.1, 0.3, 0.5],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "optimizer": {
        "lr": [1e-4, 1e-3, 1e-2],
        "weight_decay": [0, 1e-5, 1e-4],
    },
    "training_trick": {
        "warmup_steps": [100, 500, 1000],
        "T_max": [50, 100, 200],
    },
    "data_augmentation": {
        "crop_padding": [2, 4, 8],
        "mixup_alpha": [0.1, 0.2, 0.4],
    },
    "architecture": {
        "hidden_dim": [128, 256, 512],
        "num_layers": [2, 4, 6],
    },
}


@dataclass
class HyperparamVariant:
    """A technique with specific hyperparameter values."""

    technique: Technique
    params: Dict[str, Any]
    variant_id: str


class HyperparamSearch:
    """Generate hyperparameter variants for techniques."""

    def __init__(self, param_grids: Dict[str, Dict[str, List[Any]]] | None = None):
        self._grids = param_grids or PARAM_GRIDS

    def generate_variants(self, technique: Technique) -> List[HyperparamVariant]:
        """Generate parameter variants for a technique based on its category."""
        grid = self._grids.get(technique.category, {})
        if not grid:
            return [
                HyperparamVariant(
                    technique=technique,
                    params={},
                    variant_id=f"{technique.name}_default",
                )
            ]

        variants = []
        # Generate one variant per parameter value (not full grid search)
        param_names = list(grid.keys())
        if not param_names:
            return [
                HyperparamVariant(
                    technique=technique,
                    params={},
                    variant_id=f"{technique.name}_default",
                )
            ]

        # Use first param as primary sweep dimension
        primary = param_names[0]
        defaults = {k: v[0] for k, v in grid.items()}

        for i, val in enumerate(grid[primary]):
            params = dict(defaults)
            params[primary] = val
            variants.append(
                HyperparamVariant(
                    technique=technique,
                    params=params,
                    variant_id=f"{technique.name}_v{i}",
                )
            )

        return variants
