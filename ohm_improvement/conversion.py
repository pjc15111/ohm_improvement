from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.simulator.tiles.functions import AnalogFunction

from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
from torch import Tensor
import torch.nn as nn

from typing import Optional, Tuple, Int
from copy import deepcopy


class TileHolder():
    def __init__(self, tile):
        self.tile = tile


class OITile(nn.Module):
    def __init__(
        self,
        tile,
        n_realizations: Int,
        n_times: Int,
        g_converter: Optional[BaseConductanceConverter],
    ):
        nn.Module.__init__(self)

        if tile.__class__ not in {InferenceTile}:
            raise TypeError("OITile only supports conversion of InferenceTile")

        # probably want to initialize weights using a Converter

        self.train(tile.training)

        self.original_drift_compensation = deepcopy(
            tile.rpu_config.drift_compensation
        )
        self.compensation = convert_compensation(
            self.original_drift_compensation
        )
        tile.rpu_config.drift_compensation = None

        self.out_scaling_alpha = None
        if tile.out_scaling_alpha is not None:
            self.out_scaling_alpha = nn.Parameter(
                tile.out_scaling_alpha.detach().clone(),
                requires_grad=tile.out_scaling_alpha.requires_grad
            )

        self.digital_bias = tile.digital_bias
        if self.digital_bias:
            self.bias = nn.Parameter(
                tile.bias.detach().clone(),
                requires_grad=tile.bias.requires_grad
            )

        self.tile = TileHolder(tile)

    def program_weights(self, prog_noise, drift_noise, read_noise):
        pass

    def apply_out_scaling(self, values, tensor_view):
        if self.out_scaling_alpha is not None:
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

    def analog_forward(self, x_input):
        return AnalogFunction.apply(
            self.tile.tile.get_analog_ctx(),
            self.tile.tile,
            x_input,
            self.tile.tile.shared_weights,
            not self.training
        )

    def drift_weights(self, noise):
        # TODO: Apply programming noise, drift noise

        self.reference = self.compensation.init_baseline(self.tile.tile)

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple[int, ...]] = None
    ) -> Tensor:
        probe = self.compensation.get_readout_tensor(
            self.tile.tile.in_size
        ).to(self.tile.tile.shared_weights)
        probe_output = self.analog_forward(self, probe)

        output = self.analog_forward(x_input)
        output = self.compensation.apply(
            probe_output, self.reference, output
        )

        if tensor_view is None:
            tensor_view = self.tile.tile.get_tensor_view(output.dim())
        out = self.apply_out_scaling(output, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    def to_analog(self, device):
        pass


def convert_to_oi(
    module: nn.Module,
    n_realizations: Int,
    n_times: Int,
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
