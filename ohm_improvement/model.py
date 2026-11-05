from aihwkit.inference.converter.base import BaseConductanceConverter

from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
import torch.nn as nn
import torch

from typing import Optional, List
from copy import deepcopy

from ohm_improvement.tiles import OITile


def convert_to_oi(
    module: nn.Module,
    n_realizations: int,
    t_inferences: List[float],
    inplace: bool = False,
    g_converter: Optional[BaseConductanceConverter] = None,
) -> nn.Module:
    CONVERSION_MAP = {
        InferenceTile: OITile
    }

    if not inplace:
        module = deepcopy(module)

    if module.__class__ in CONVERSION_MAP:
        module = CONVERSION_MAP[module.__class__](
            module,
            n_realizations,
            t_inferences,
            g_converter
        )

    # named_children() suppresses duplicate references, returning each
    # child only once. So duplicate references will wind up referring
    # to different objects. But we only get to this method after going
    # through  convert_to_analog first, which has the same bug, so we
    # do no additional harm.
    for name, mod in module.named_children():
        module._modules[name] = convert_to_oi(
            mod,
            n_realizations,
            t_inferences,
            inplace=True,
            g_converter=g_converter
        )

    return module


def oi_to_analog(model, inplace=False):
    if not inplace:
        memo = {
            id(module): module.to_analog()
            for module in model.modules()
            if isinstance(module, OITile)
        }
        return deepcopy(model, memo)
        model = deepcopy(model)

    if isinstance(model, OITile):
        return model.to_analog()

    for name, child in model.named_children():
        if child is not None:
            model._modules[name] = oi_to_analog(child, inplace=True)

    return model


def program_weights(
    model: nn.Module,
    realization_index: int,
    time_index: int
) -> None:
    """Apply one realization and timestamp to every OITile."""
    for module in model.modules():
        if isinstance(module, OITile):
            module.program_weights(
                realization_index,
                time_index
            )


@torch.no_grad()
def clamp_conductances(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, OITile):
            module.clamp_conductances()


@torch.no_grad()
def clamp_significance(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, OITile):
            module.clamp_significance()


def clamp_model(model: nn.Module) -> None:
    clamp_conductances(model)
    clamp_significance(model)


def optimization_parameters(
    model,
    include_conductances=True,
    include_output_scaling=True,
    include_bias=False,
    include_significance=True
):
    parameters = []

    for tile in model.modules():
        if not isinstance(tile, OITile):
            continue

        if include_conductances:
            parameters.extend(tile.g_plus)
            parameters.extend(tile.g_minus)

        if include_output_scaling and tile.out_scaling_alpha is not None:
            parameters.append(tile.out_scaling_alpha)

        if include_bias and tile.digital_bias:
            parameters.append(tile.bias)

        if include_significance and tile.significance:
            parameters.append(tile.significance)

    return parameters
