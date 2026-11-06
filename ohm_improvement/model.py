from aihwkit.inference.converter.base import BaseConductanceConverter

from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
import hashlib
import math

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
    noise_seeds=None,
) -> nn.Module:
    if noise_seeds is None:
        noise_seeds = dict(zip(
            ("train", "validation", "evaluation"),
            torch.randint(0, 2**63 - 1, (3,), dtype=torch.int64).tolist(),
        ))
    elif (set(noise_seeds) != {"train", "validation", "evaluation"}
            or any(type(s) is not int or not 0 <= s < 2**63
                   for s in noise_seeds.values())
            or len(set(noise_seeds.values())) != 3):
        raise ValueError("noise_seeds requires three distinct integer seeds in [0, 2**63)")

    if not inplace:
        module = deepcopy(module)

    # Use stable module paths so same-shaped tiles have independent streams.
    def convert(child, path):
        if child.__class__ is InferenceTile:
            tile_seeds = {
                name: int.from_bytes(hashlib.blake2b(
                    f"{seed}:{path}".encode(), digest_size=8
                ).digest(), "little") % (2**63)
                for name, seed in noise_seeds.items()
            }
            return OITile(child, n_realizations, t_inferences, g_converter,
                          noise_seeds=tile_seeds)
        for name, grandchild in child.named_children():
            child._modules[name] = convert(grandchild, f"{path}.{name}")
        return child

    module = convert(module, "")

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
    time_index: int,
    noise_split: str = "train",
) -> None:
    """Apply one realization and timestamp to every OITile."""
    for module in model.modules():
        if isinstance(module, OITile):
            module.program_weights(
                realization_index,
                time_index,
                noise_split=noise_split,
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
    model: nn.Module,
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

    for tile in model.modules():
        if not isinstance(tile, OITile):
            continue
        if include_conductances:
            groups["conductances"]["params"].extend(tile.g_plus)
            groups["conductances"]["params"].extend(tile.g_minus)
        if include_significance and tile.significance is not None:
            groups["significance"]["params"].append(tile.significance)
        if include_bias and tile.digital_bias:
            groups["bias"]["params"].append(tile.bias)
        if include_output_scaling and tile.out_scaling_alpha is not None:
            groups["output_scaling"]["params"].append(tile.out_scaling_alpha)

    result = []
    for name, group in groups.items():
        parameters = [p for p in group["params"] if p.requires_grad]
        if parameters:
            result.append(dict(name=name, params=parameters, lr=group["lr"]))
    return result
