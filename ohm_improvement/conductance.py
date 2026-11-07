"""Conversion between logical weights and differential conductance pairs."""

from dataclasses import dataclass
import torch
from torch import Tensor
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import CustomPairConductanceConverter


@dataclass
class ConductanceMapping:
    g_plus: list[Tensor]
    g_minus: list[Tensor]
    significance: Tensor  # Shape: [n_pairs]
    scale: Tensor        # Scalar

    def validate(self) -> None:
        """Reject malformed mappings before tensor broadcasting can hide errors."""
        count = len(self.g_plus)
        if not count or len(self.g_minus) != count or self.significance.shape != (count,):
            raise ValueError("Expected matching, nonempty differential pairs and factors")
        reference = self.g_plus[0]
        if any(g.shape != reference.shape for g in [*self.g_plus, *self.g_minus]):
            raise ValueError("All conductance tensors must have the same shape")
        tensors = [*self.g_plus, *self.g_minus, self.significance, self.scale]
        if any(t.device != reference.device or t.dtype != reference.dtype for t in tensors):
            raise ValueError("Conductances, significance, and scale must share device and dtype")
        if self.scale.ndim != 0 or not torch.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Conversion scale must be a finite, positive scalar")


def convert_to_conductances(
    converter: BaseConductanceConverter,
    weights: Tensor
) -> ConductanceMapping:
    # AIHWKIT's custom converter interpolates through NumPy and currently mixes
    # CPU linspace with CUDA scalars. Initialization can run on CPU once.
    conversion_weights = weights.cpu() if isinstance(
        converter, CustomPairConductanceConverter
    ) else weights
    conductances, params = converter.convert_to_conductances(conversion_weights)
    conductances = [g.to(weights) for g in conductances]

    mapping = ConductanceMapping(
        g_plus=conductances[::2],
        g_minus=conductances[1::2],
        significance=weights.new_tensor(params.get("f_lst", [1.0])),
        scale=params["scale_ratio"].detach().to(weights).clone(),
    )
    mapping.validate()
    if mapping.g_plus[0].shape != weights.shape:
        raise ValueError("Converted conductances must preserve the weight tensor shape")
    return mapping


def convert_back_to_weights(mapping: ConductanceMapping) -> Tensor:
    """Reconstruct logical weights from conductances and their scale."""
    mapping.validate()
    return _reconstruct_weights(mapping)


def _reconstruct_weights(mapping: ConductanceMapping) -> Tensor:
    """Reconstruct an internally produced mapping without boundary validation."""
    weights = torch.zeros_like(mapping.g_plus[0])

    for factor, g_plus, g_minus in zip(
        mapping.significance,
        mapping.g_plus,
        mapping.g_minus,
    ):
        weights = weights + factor * (g_plus - g_minus)

    return weights / mapping.scale
