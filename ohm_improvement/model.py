from aihwkit.inference.converter.base import BaseConductanceConverter

from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
import torch.nn as nn

from typing import Optional
from copy import deepcopy

from ohm_improvement.tiles import OITile


def convert_to_oi(
    module: nn.Module,
    n_realizations: int,
    n_times: int,
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
            n_times,
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
            n_times,
            inplace=True,
            g_converter=g_converter
        )

    return module


def program_weights(
    model: nn.Module,
    realization_index: int,
    time_index: int,
    time: float,
) -> None:
    """Apply one realization and timestamp to every OITile."""
    for module in model.modules():
        if isinstance(module, OITile):
            module.program_weights(
                realization_index,
                time_index,
                time,
            )
