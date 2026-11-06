from aihwkit.simulator.tiles.inference import (
    InferenceTile
)
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.simulator.tiles.functions import AnalogFunction

import hashlib

import torch.nn as nn

import torch

from torch import Tensor
from typing import Optional, Tuple, List
from copy import deepcopy

from .compensation import convert_compensation
from .conductance import (
    convert_to_conductances,
    convert_back_to_weights,
    ConductanceMapping
)
from .noise import convert_noise_model


class TileHolder():
    def __init__(self, tile):
        self.tile = tile


class OITile(nn.Module):
    def __init__(
        self,
        tile,
        n_realizations: int,
        t_inferences: List[float],
        g_converter: Optional[BaseConductanceConverter],
        train_significance = True,
        *,
        noise_seeds,
    ):
        nn.Module.__init__(self)

        self.n_realizations = n_realizations
        self.t_inferences = t_inferences

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

        self.tile = TileHolder(tile)
        self.init_conductances(train_significance, g_converter)
        self.init_noise(n_realizations, len(t_inferences), noise_seeds)

    def init_conductances(
        self,
        train_significance,
        g_converter: Optional[BaseConductanceConverter] = None,
    ) -> None:
        tile = self.tile.tile

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

    def init_noise(self, n_realizations: int, n_times: int, noise_seeds) -> None:
        """Store seeds already initialized and validated by convert_to_oi.

        Direct OITile callers must supply valid train/validation/evaluation seeds.
        """
        if n_realizations < 1 or n_times < 1:
            raise ValueError("n_realizations and n_times must be positive")
        self.n_realizations = n_realizations
        self.register_buffer("noise_seeds", torch.tensor(
            [noise_seeds[name] for name in ("train", "validation", "evaluation")],
            dtype=torch.int64,
        ))

    def noise_sample(self, kind, pair, side, realization, time_index=0,
                     noise_split="train"):
        """Return fixed randomness without touching the global RNG.

        Programming and drift samples are shared across timestamps. Read samples
        are keyed by elapsed time, independent of evaluation order or time grid.
        Reproducibility is within a device/backend and dtype, not across CPU/CUDA.
        """
        names = ("train", "validation", "evaluation")
        if noise_split not in names:
            raise ValueError(f"Unknown noise split: {noise_split}")
        if kind not in ("program", "drift", "read"):
            raise ValueError(f"Unknown noise kind: {kind}")
        seed = int(self.noise_seeds[names.index(noise_split)].item())
        timestamp = float(self.t_inferences[time_index]).hex() if kind == "read" else ""
        key = f"{seed}:{kind}:{pair}:{side}:{realization}:{timestamp}"
        derived = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
        g = self.g_plus[pair]
        generator = torch.Generator(device=g.device).manual_seed(derived)
        return torch.randn(g.shape, generator=generator, device=g.device, dtype=g.dtype)

    def apply_out_scaling(self, values, tensor_view):
        if self.out_scaling_alpha is not None:
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

    def analog_forward(self, x_input):
        if self.active_weights is None:
            raise RuntimeError("Call program_weights() before forward()")

        tile = self.tile.tile
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

    def program_weights(
        self,
        realization_index: int,
        time_index: int,
        noise_split: str = "train",
    ) -> None:
        if not 0 <= realization_index < self.n_realizations:
            raise IndexError("realization_index out of range")
        if not 0 <= time_index < len(self.t_inferences):
            raise IndexError("time_index out of range")
        if noise_split not in ("train", "validation", "evaluation"):
            raise ValueError(f"Unknown noise split: {noise_split}")
        time = self.t_inferences[time_index]
        if time < 0:
            raise ValueError("time must be nonnegative")

        r = realization_index
        t = time_index

        programmed = ([], [])
        drift_coefficients = ([], [])

        for side, targets in enumerate((self.g_plus, self.g_minus)):
            for p, g_target in enumerate(targets):
                programmed[side].append(
                    self.noise_model.apply_programming_noise_to_conductance(
                        g_target,
                        self.noise_sample("program", p, side, r, t, noise_split),
                    )
                )
                drift_coefficients[side].append(
                    self.noise_model.generate_drift_coefficients(
                        g_target,
                        self.noise_sample("drift", p, side, r, t, noise_split),
                    )
                )

        # Compensation reference: programmed state, before drift.
        if self.compensation is not None:
            self.install_conductances(*programmed)
            self.reference = self.compensation.init_baseline(
                self.tile.tile,
                self.active_weights
            )

        drifted = ([], [])

        for side in range(2):
            for p, (g_prog, nu) in enumerate(
                    zip(programmed[side], drift_coefficients[side])
            ):
                drifted[side].append(
                    self.noise_model.apply_drift_noise_to_conductance(
                        g_prog,
                        nu,
                        time,
                        self.noise_sample("read", p, side, r, t, noise_split),
                    )
                )

        self.install_conductances(*drifted)

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple[int, ...]] = None
    ) -> Tensor:

        output = self.analog_forward(x_input)
        if self.compensation is not None:
            probe = self.compensation.get_readout_tensor(
                self.tile.tile.in_size
            ).to(self.active_weights)
            probe_output = self.analog_forward(probe)
            output = self.compensation.apply(
                probe_output, self.reference, output
            )

        if tensor_view is None:
            tensor_view = self.tile.tile.get_tensor_view(output.dim())
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
        if self.active_weights is None:
            raise RuntimeError("Call program_weights() before to_analog()")

        tile = deepcopy(self.tile.tile)

        # Independent storage, registered for ordinary PyTorch handling.
        tile.shared_weights = nn.Parameter(
            self.active_weights.detach().clone(),
            requires_grad=False,
        )

        with torch.no_grad():
            if self.compensation is not None:
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
