"""Model-level interface for differentiable conductance optimization."""

from copy import deepcopy
import hashlib
import math
from typing import TypeAlias

import torch
from torch import nn
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.nn.modules.conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
from aihwkit.nn.modules.conv_mapped import (
    AnalogConv1dMapped, AnalogConv2dMapped, AnalogConv3dMapped,
)
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.simulator.tiles.inference import InferenceTile

from .tiles import OITile

# AIHWKIT's common base covers AnalogWrapper, AnalogSequential, and standalone
# analog layers (e.g. AnalogLinear), without accepting arbitrary digital modules.
AnalogModel: TypeAlias = AnalogLayerBase


class OIModel(nn.Module):
    def __init__(
        self, analog_model: AnalogModel, *, seed: int | None = None,
        g_converter: BaseConductanceConverter | None = None,
        inplace: bool = False, train_significance: bool = True,
    ):
        super().__init__()
        if not isinstance(analog_model, AnalogLayerBase):
            raise TypeError("OIModel requires an AIHWKIT analog model or layer")
        for layer in analog_model.modules():
            if isinstance(layer, (AnalogConv1d, AnalogConv3d, AnalogConv1dMapped, AnalogConv3dMapped)):
                raise NotImplementedError("OI supports unfolded Conv2d only; indexed convolutions are unsupported")
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, ()).item()
        elif type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        model = analog_model if inplace else deepcopy(analog_model)
        converted = {}

        def convert(child, path):
            if id(child) in converted:
                return converted[id(child)]
            if type(child) is InferenceTile:
                tile_seed = int.from_bytes(hashlib.blake2b(
                    f"{seed}:{path}".encode(), digest_size=8,
                ).digest(), "little") % (2**63)
                result = OITile(child, g_converter, train_significance, seed=tile_seed)
                converted[id(child)] = result
                return result
            converted[id(child)] = child
            if isinstance(child, (AnalogConv2d, AnalogConv2dMapped)):
                child.use_indexed = False
            for name, grandchild in list(child._modules.items()):
                if grandchild is not None:
                    child._modules[name] = convert(grandchild, f"{path}.{name}")
            return child

        self._model = convert(model, "")
        self.training = analog_model.training
        if not any(self.oi_tiles()):
            raise ValueError("The analog model contains no supported InferenceTile")

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def named_oi_tiles(self):
        """Yield tile names relative to the retained original analog model."""
        for name, tile in self._model.named_modules():
            if isinstance(tile, OITile):
                yield name, tile

    def oi_tiles(self):
        for _, tile in self.named_oi_tiles():
            yield tile

    def program(self, realization_id: str) -> None:
        """Replace the current programmed realization on every tile."""
        if not isinstance(realization_id, str):
            raise TypeError("realization_id must be a string")
        for tile in self.oi_tiles():
            tile.program(realization_id)

    def drift(self, t_inference: float) -> None:
        """Drift the current realization to elapsed seconds since programming."""
        for tile in self.oi_tiles():
            tile.drift(t_inference)

    def to_analog(self) -> AnalogModel:
        """Export an independent stock model at the current programmed/drifted state."""
        memo = {id(tile): tile.to_analog() for tile in self.oi_tiles()}
        return deepcopy(self._model, memo)

    def clamp_conductances(self) -> None:
        for tile in self.oi_tiles():
            tile.clamp_conductances()

    def clamp_significance(self) -> None:
        for tile in self.oi_tiles():
            tile.clamp_significance()

    def clamp(self) -> None:
        self.clamp_conductances()
        self.clamp_significance()

    def optimization_parameters(
        self,
        include_conductances=True,
        include_output_scaling=True,
        include_bias=False,
        include_significance=True,
        *,
        conductance_lr: float = 0.01,
        significance_lr: float = 0.0001,
        bias_lr: float = 0.0001,
        output_scaling_lr: float = 0.0001,
    ) -> list[dict]:
        """Return named parameter groups ready for torch.optim.Adam.

        Each group has its own learning rate; passing lr to Adam does not override
        these explicit rates. Use the keyword overrides here instead. Excluded,
        absent, and frozen parameters are omitted, as are empty groups.
        """
        groups = {
            "conductances": dict(params=[], lr=conductance_lr),
            "significance": dict(params=[], lr=significance_lr),
            "bias": dict(params=[], lr=bias_lr),
            "output_scaling": dict(params=[], lr=output_scaling_lr),
        }
        for name, group in groups.items():
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError(f"{name} learning rate must be finite and nonnegative")

        for tile in self.oi_tiles():
            if include_conductances:
                groups["conductances"]["params"].extend(tile.g_plus)
                groups["conductances"]["params"].extend(tile.g_minus)
            if include_significance and tile.significance is not None:
                groups["significance"]["params"].append(tile.significance)
            if include_bias and tile.digital_bias:
                groups["bias"]["params"].append(tile.bias)
            if include_output_scaling and tile.out_scaling_alpha is not None:
                groups["output_scaling"]["params"].append(tile.out_scaling_alpha)

        if include_bias:
            # Mapped arrays own one digital bias after the partial tile sums.
            for module in self._model.modules():
                if isinstance(module, (TileModuleArray, AnalogConv2dMapped)) and module.bias is not None:
                    groups["bias"]["params"].append(module.bias)

        result = []
        for name, group in groups.items():
            parameters = [p for p in group["params"] if p.requires_grad]
            if parameters:
                result.append(dict(name=name, params=parameters, lr=group["lr"]))
        return result
