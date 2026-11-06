from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.simulator.tiles.functions import AnalogFunction

import hashlib
import json
import math

import torch.nn as nn

import torch

from torch import Tensor
from typing import Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass

from .compensation import convert_compensation
from .conductance import (
    convert_to_conductances,
    convert_back_to_weights,
    ConductanceMapping
)
from .noise import convert_noise_model


@dataclass
class AnalogTileHolder:
    """Hold the backend tile without registering it as a PyTorch submodule."""

    analog_tile: InferenceTile


class OITile(nn.Module):
    def __init__(
        self,
        tile,
        g_converter: Optional[BaseConductanceConverter] = None,
        train_significance = True,
        *,
        seed: int,
    ):
        nn.Module.__init__(self)

        if tile.__class__ not in {InferenceTile}:
            raise TypeError("OITile only supports conversion of InferenceTile")

        self.train(tile.training)

        self.original_drift_compensation = deepcopy(
            tile.rpu_config.drift_compensation
        )
        self.compensation = convert_compensation(
            self.original_drift_compensation
        )
        tile.rpu_config.drift_compensation = None

        if not hasattr(tile.rpu_config, "noise_model"):
            raise Exception("Seems that RPUConfig is not of type InferenceRPUConfig.")
        self.noise_model = convert_noise_model(tile.rpu_config.noise_model)

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

        self.analog_tile_holder = AnalogTileHolder(tile)
        self.init_conductances(train_significance, g_converter)
        # Model construction initializes/validates seeds; direct callers supply one.
        self.register_buffer("noise_seed", torch.tensor(seed, dtype=torch.int64))
        self._programmed = None
        self._realization_id = None
        self._parameter_versions = []
        self._drifted = False

    def get_analog_tile(self) -> InferenceTile:
        """Return the underlying AIHWKIT analog tile."""
        return self.analog_tile_holder.analog_tile

    def init_conductances(
        self,
        train_significance,
        g_converter: Optional[BaseConductanceConverter] = None,
    ) -> None:
        tile = self.get_analog_tile()

        if tile.shared_weights is None:
            raise ValueError("OITile requires a tile with shared_weights")

        if g_converter is None:
            g_converter = tile.rpu_config.noise_model.g_converter

        self.g_converter = deepcopy(g_converter)

        # Read raw tile weights, without mapping or learned output scaling.
        weights, _ = tile.get_weights(apply_weight_scaling=False)
        weights = weights.detach().to(
            device=tile.shared_weights.device,
            dtype=tile.shared_weights.dtype,
        )

        # get_weights() returns [out_features, in_features].
        # Preserve the native layout expected by our installation code.
        if tile.shared_weights.is_cuda:
            weights = weights.T

        weights = weights.contiguous().clone()

        if "shared_weights" in tile._parameters:
            del tile._parameters["shared_weights"]

        # Mutable backend holder, never passed as the autograd weight argument.
        tile.shared_weights = weights.detach().clone()
        # CUDA's first shared-weight binding copies backend weights into the
        # supplied tensor. Complete it using the disposable holder, before any
        # differentiable programmed/drifted weights are passed to AnalogFunction.
        tile.ensure_shared_weights()

        # Set by install_conductances().
        self.active_weights = None

        with torch.no_grad():
            mapping = convert_to_conductances(self.g_converter, weights)

        self.g_plus = nn.ParameterList(
            [nn.Parameter(g.detach().clone()) for g in mapping.g_plus]
        )
        self.g_minus = nn.ParameterList(
            [nn.Parameter(g.detach().clone()) for g in mapping.g_minus]
        )
        self.significance = nn.Parameter(
            mapping.significance.detach().clone(),
            requires_grad=train_significance,
        )
        self.register_buffer("scale", mapping.scale.detach().clone())

    def noise_sample(self, kind: str, pair: int, side: str,
                     realization_id: str, t_inference: float = 0.) -> Tensor:
        """Regenerate a sample without changing the global RNG.

        Programming/drift samples omit time; read samples are keyed by elapsed
        time. Reproducibility is within a backend/device/dtype, not across them.
        """
        if side not in ("plus", "minus"):
            raise ValueError("side must be 'plus' or 'minus'")
        if kind not in ("program", "drift", "read"):
            raise ValueError(f"Unknown noise kind: {kind}")
        if not isinstance(realization_id, str):
            raise TypeError("realization_id must be a string")
        timestamp = float(t_inference or 0.).hex() if kind == "read" else None
        key = json.dumps([self.noise_seed.item(), kind, pair, side, realization_id, timestamp])
        derived = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
        g = self.g_plus[pair]
        generator = torch.Generator(device=g.device).manual_seed(derived)
        return torch.randn(g.shape, generator=generator, device=g.device, dtype=g.dtype)

    def _require_programmed(self):
        if self._programmed is None:
            raise RuntimeError("Call program() first")
        if any(p._version != version for p, version in self._parameter_versions):
            raise RuntimeError("Parameters changed; call program() again")

    def apply_out_scaling(self, values, tensor_view):
        if self.out_scaling_alpha is not None:
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

    def analog_forward(self, x_input):
        self._require_programmed()

        tile = self.get_analog_tile()
        return AnalogFunction.apply(
            tile.get_analog_ctx(),
            tile,
            x_input,
            self.active_weights,
            not self.training,
        )

    def install_conductances(self, gp, gm):
        self.active_weights = convert_back_to_weights(
            ConductanceMapping(
                g_plus=gp,
                g_minus=gm,
                significance=self.significance,
                scale=self.scale,
            )
        ).contiguous()

    def program(self, realization_id: str) -> None:
        """Retain one programmed realization and its drift coefficients.

        This builds an autograd graph. For multiple backward calls sharing this
        state, retain_graph=True is required until the last call. Reprogram after
        backward to start a new graph, and after any parameter update.
        """
        if not isinstance(realization_id, str):
            raise TypeError("realization_id must be a string")
        self._programmed = None
        self.active_weights = None
        self._drifted = False
        programmed, coefficients = [], []
        for side, targets in (("plus", self.g_plus), ("minus", self.g_minus)):
            gp, nu = [], []
            for pair, target in enumerate(targets):
                gp.append(self.noise_model.apply_programming_noise_to_conductance(
                    target, self.noise_sample("program", pair, side, realization_id)))
                nu.append(self.noise_model.generate_drift_coefficients(
                    target, self.noise_sample("drift", pair, side, realization_id)))
            programmed.append(gp)
            coefficients.append(nu)
        self.install_conductances(*programmed)
        if self.compensation is not None:
            self.reference = self.compensation.init_baseline(
                self.get_analog_tile(), self.active_weights)
        self._programmed = (programmed, coefficients)
        self._realization_id = realization_id
        self._parameter_versions = [(p, p._version) for p in self.parameters()]

    def drift(self, t_inference: float) -> None:
        """Apply drift/read noise from the retained state, without compounding."""
        self._require_programmed()
        t_inference = float(t_inference)
        if not math.isfinite(t_inference) or t_inference < 0:
            raise ValueError("t_inference must be finite and nonnegative")
        programmed, coefficients = self._programmed
        drifted = []
        for side, gs, nus in zip(("plus", "minus"), programmed, coefficients):
            drifted.append([
                self.noise_model.apply_drift_noise_to_conductance(
                    g, nu, t_inference,
                    self.noise_sample("read", pair, side, self._realization_id, t_inference))
                for pair, (g, nu) in enumerate(zip(gs, nus))
            ])
        self.install_conductances(*drifted)
        self._drifted = True

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple[int, ...]] = None
    ) -> Tensor:

        output = self.analog_forward(x_input)
        if self.compensation is not None and self._drifted:
            probe = self.compensation.get_readout_tensor(
                self.get_analog_tile().in_size
            ).to(self.active_weights)
            probe_output = self.analog_forward(probe)
            output = self.compensation.apply(
                probe_output, self.reference, output
            )

        if tensor_view is None:
            tensor_view = self.get_analog_tile().get_tensor_view(output.dim())
        out = self.apply_out_scaling(output, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    @torch.no_grad()
    def clamp_conductances(self) -> None:
        for side in (self.g_plus, self.g_minus):
            for g in side:
                g.clamp_(self.g_converter.g_min, self.g_converter.g_max)

    @torch.no_grad()
    def clamp_significance(self):
        self.significance.clamp_(0.1, 5.0)

    def to_analog(self):
        self._require_programmed()

        tile = deepcopy(self.get_analog_tile())
        # Bind the copy's disposable holder before supplying snapshot weights;
        # its first binding may otherwise overwrite them with backend state.
        tile.ensure_shared_weights()

        # Independent storage, registered for ordinary PyTorch handling.
        tile.shared_weights = nn.Parameter(
            self.active_weights.detach().clone(),
            requires_grad=False,
        )

        with torch.no_grad():
            tile.alpha = torch.ones_like(tile.alpha)
            if self.compensation is not None and self._drifted:
                probe = self.compensation.get_readout_tensor(
                    tile.in_size
                ).to(tile.shared_weights)

                # Stock compensation is still disabled here.
                probe_output = AnalogFunction.apply(
                    tile.get_analog_ctx(),
                    tile,
                    probe,
                    tile.shared_weights,
                    True,
                )

                current_readout = self.compensation.readout(probe_output)
                tile.alpha = (
                    self.reference / current_readout
                ).detach().clone()
                tile.drift_baseline = self.reference.detach().clone()

            # Bind the exported weights even when compensation is disabled.
            tile.ensure_shared_weights(tile.shared_weights)

        tile.rpu_config.drift_compensation = deepcopy(
            self.original_drift_compensation
        )

        tile.rpu_config.noise_model.g_converter.f_lst = (
            self.significance.detach().cpu().tolist()
        )

        if self.out_scaling_alpha is not None:
            tile.out_scaling_alpha = nn.Parameter(
                self.out_scaling_alpha.detach().clone(),
                requires_grad=self.out_scaling_alpha.requires_grad,
            )
        else:
            tile.out_scaling_alpha = None

        if self.digital_bias:
            tile.bias = nn.Parameter(
                self.bias.detach().clone(),
                requires_grad=self.bias.requires_grad,
            )

        tile.eval()
        return tile
