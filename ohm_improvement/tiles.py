from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.simulator.tiles.functions import AnalogFunction

import torch.nn as nn

import torch

from torch import Tensor
from typing import Optional, Tuple
from copy import deepcopy

from .compensation import convert_compensation
from .conductance import convert_to_conductances, convert_back_to_weights


class TileHolder():
    def __init__(self, tile):
        self.tile = tile


class OITile(nn.Module):
    def __init__(
        self,
        tile,
        n_realizations: int,
        n_times: int,
        g_converter: Optional[BaseConductanceConverter],
    ):
        nn.Module.__init__(self)

        self.n_realizations = n_realizations
        self.n_times = n_times

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
        self.init_conductances(g_converter)
        self.init_noise(n_realizations, n_times)

    def init_conductances(
            self,
            g_converter: Optional[BaseConductanceConverter] = None,
    ) -> None:
        tile = self.tile.tile

        if tile.shared_weights is None:
            raise ValueError("OITile requires a tile with shared_weights")

        if g_converter is None:
            g_converter = tile.rpu_config.noise_model.g_converter

        self.g_converter = deepcopy(g_converter)

        # Keep the tile's native weight layout, including CUDA's transpose.
        # These are raw tile weights: mapping/output scales remain separate.
        weights = tile.shared_weights.detach().clone()

        if "shared_weights" in tile._parameters:
            del tile._parameters["shared_weights"]
        tile.shared_weights = None

        with torch.no_grad():
            (g_plus, g_minus), scale = convert_to_conductances(
                self.g_converter, weights
            )

        # TODO: Add trainable significance per weight

        self.g_plus = nn.ParameterList(
            [nn.Parameter(g.detach().clone()) for g in g_plus]
        )
        self.g_minus = nn.ParameterList(
            [nn.Parameter(g.detach().clone()) for g in g_minus]
        )
        self.register_buffer("scale", scale.detach().clone())

    def init_noise(self, n_realizations: int, n_times: int) -> None:
        """Sample fixed randomness for programming, drift, and read noise."""
        if n_realizations < 1 or n_times < 1:
            raise ValueError("n_realizations and n_times must be positive")

        self.n_realizations = n_realizations
        self.n_times = n_times

        n_pairs = len(self.g_plus)
        g = self.g_plus[0]

        # Dimensions: [pair, side, realization, *conductance_shape].
        # Side 0 is positive; side 1 is negative.
        shape = (n_pairs, 2, n_realizations, *g.shape)

        self.register_buffer(
            "prog_noise",
            torch.randn(shape, device=g.device, dtype=g.dtype),
        )
        self.register_buffer(
            "drift_noise",
            torch.randn(shape, device=g.device, dtype=g.dtype),
        )

        # Read noise is independent at each timestamp.
        # Dimensions: [pair, side, realization, time, *conductance_shape].
        read_shape = (n_pairs, 2, n_realizations, n_times, *g.shape)

        self.register_buffer(
            "read_noise",
            torch.randn(read_shape, device=g.device, dtype=g.dtype),
        )

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

    def install_conductances(self, gp, gm):
        weights = convert_back_to_weights(
            self.g_converter,
            (gp, gm),
            self.scale
        )

        self.tile.tile.shared_weights = weights.contiguous()

    def program_weights(
        self,
        realization_index: int,
        time_index: int,
        time: float,
    ) -> None:
        if not 0 <= realization_index < self.n_realizations:
            raise IndexError("realization_index out of range")
        if not 0 <= time_index < self.n_times:
            raise IndexError("time_index out of range")
        if time < 0:
            raise ValueError("time must be nonnegative")

        r = realization_index
        t = time_index

        programmed = ([], [])
        drift_coefficients = ([], [])

        for side, targets in enumerate((self.g_plus, self.g_minus)):
            for p, g_target in enumerate(targets):
                programmed[side].append(
                    self.noise.apply_programming_noise_to_conductance(
                        g_target,
                        self.prog_noise[p, side, r],
                    )
                )
                drift_coefficients[side].append(
                    self.noise.generate_drift_coefficients(
                        g_target,
                        self.drift_noise[p, side, r],
                    )
                )

        # Compensation reference: programmed state, before drift.
        if self.compensation is not None:
            self.install_conductances(*programmed)
            self.reference = self.compensation.init_baseline(self.tile.tile)

        drifted = ([], [])

        for side in range(2):
            for p, (g_prog, nu) in enumerate(
                    zip(programmed[side], drift_coefficients[side])
            ):
                drifted[side].append(
                    self.noise.apply_drift_noise_to_conductance(
                        g_prog,
                        nu,
                        time,
                        self.read_noise[p, side, r, t],
                    )
                )

        self.install_conductances(*drifted)

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple[int, ...]] = None
    ) -> Tensor:
        probe = self.compensation.get_readout_tensor(
            self.tile.tile.in_size
        ).to(self.tile.tile.shared_weights)
        probe_output = self.analog_forward(probe)

        output = self.analog_forward(x_input)
        if self.compensation is not None:
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
