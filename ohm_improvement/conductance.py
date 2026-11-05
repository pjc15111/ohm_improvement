from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
    DualPairConductanceConverter,
    NPairConductanceConverter,
    CustomPairConductanceConverter
)
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


def npair_to_conductances(
    weight: Tensor,
    g_min: float,
    g_max: float,
    factors: List[float],
) -> ConductanceMapping:
    conds, params = NPairConductanceConverter(
        f_lst=factors,
        g_max=g_max,
        g_min=g_min,
    ).convert_to_conductances(weight)

    return ConductanceMapping(
        g_plus=conds[::2],
        g_minus=conds[1::2],
        significance=weight.new_tensor(params["f_lst"]),
        scale=params["scale_ratio"].detach().clone(),
    )


def convert_to_conductances(
    converter: BaseConductanceConverter,
    weights: Tensor
) -> ConductanceMapping:
    converter_type = type(converter)

    if converter_type is SinglePairConductanceConverter:
        factors = [1.0]
    elif converter_type in {
        NPairConductanceConverter,
        DualPairConductanceConverter,
    }:
        factors = converter.f_lst
    elif converter_type is CustomPairConductanceConverter:
        raise NotImplementedError(
            "CustomPairConductanceConverter not supported"
        )
        # return custom_pair_to_conductances(converter, weights)
    else:
        raise NotImplementedError(
            f"Unsupported converter: {converter_type.__name__}"
        )

    return npair_to_conductances(
        weights,
        g_min=converter.g_min,
        g_max=converter.g_max,
        factors=factors,
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
