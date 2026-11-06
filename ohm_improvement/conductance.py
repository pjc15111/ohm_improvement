from aihwkit.inference.converter.base import BaseConductanceConverter

import torch
from torch import Tensor
from typing import List
from dataclasses import dataclass


@dataclass
class ConductanceMapping:
    g_plus: List[Tensor]
    g_minus: List[Tensor]
    significance: Tensor  # Shape: [n_pairs]
    scale: Tensor        # Scalar


def convert_to_conductances(
    converter: BaseConductanceConverter,
    weights: Tensor
) -> ConductanceMapping:
    conductances, params = converter.convert_to_conductances(weights)

    return ConductanceMapping(
        g_plus=conductances[::2],
        g_minus=conductances[1::2],
        significance=weights.new_tensor(params.get("f_lst", [1.0])),
        scale=params["scale_ratio"].detach().clone(),
    )


def convert_back_to_weights(mapping: ConductanceMapping) -> Tensor:
    """Reconstruct logical weights from conductances and their scale."""
    n_pairs = len(mapping.g_plus)
    if (
        n_pairs == 0
        or len(mapping.g_minus) != n_pairs
        or mapping.significance.shape != (n_pairs,)
    ):
        raise ValueError("Expected matching, nonempty differential pairs and factors")

    weights = torch.zeros_like(mapping.g_plus[0])

    for factor, g_plus, g_minus in zip(
        mapping.significance,
        mapping.g_plus,
        mapping.g_minus,
    ):
        weights = weights + factor * (g_plus - g_minus)

    return weights / mapping.scale
