from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter
)
from aihwkit.inference.converter.base import BaseConductanceConverter

from torch import Tensor

from typing import List, Tuple


def single_pair_to_conductances(
    converter: SinglePairConductanceConverter,
    weights: Tensor,
) -> Tuple[Tuple[List[Tensor], List[Tensor]], Tensor]:
    """Map logical weights to one differential pair and return its scale."""
    scale = (
        (converter.g_max - converter.g_min)
        / weights.abs().max().clamp(min=1e-7)
    )

    scaled_weights = weights * scale

    g_plus = (
        scaled_weights.clamp(min=0.0, max=converter.g_max - converter.g_min)
        + converter.g_min
    )
    g_minus = (
        (-scaled_weights).clamp(
            min=0.0, max=converter.g_max - converter.g_min
        )
        + converter.g_min
    )

    return ([g_plus], [g_minus]), scale


def single_pair_to_weights(
    converter: SinglePairConductanceConverter,
    conductances: Tuple[List[Tensor], List[Tensor]],
    scale: Tensor,
) -> Tensor:
    """Reconstruct logical weights from one differential pair."""
    g_plus, g_minus = conductances

    if len(g_plus) != 1 or len(g_minus) != 1:
        raise ValueError("Single-pair conversion requires one tensor per side.")

    return (
        (g_plus[0] - converter.g_min)
        - (g_minus[0] - converter.g_min)
    ) / scale


_CONVERTER_FUNCTIONS = {
    SinglePairConductanceConverter: (
        single_pair_to_conductances,
        single_pair_to_weights,
    ),
}


def _converter_functions(converter: BaseConductanceConverter):
    try:
        return _CONVERTER_FUNCTIONS[type(converter)]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported conductance converter: "
            f"{type(converter).__name__}"
        ) from None


def convert_to_conductances(
    converter: BaseConductanceConverter,
    weights: Tensor,
) -> Tuple[Tuple[List[Tensor], List[Tensor]], Tensor]:
    """Convert weights into positive/negative conductance lists and a scale."""
    to_conductances, _ = _converter_functions(converter)
    return to_conductances(converter, weights)


def convert_back_to_weights(
    converter: BaseConductanceConverter,
    conductances: Tuple[List[Tensor], List[Tensor]],
    scale: Tensor,
) -> Tensor:
    """Reconstruct logical weights from conductances and their scale."""
    _, to_weights = _converter_functions(converter)
    return to_weights(converter, conductances, scale)
