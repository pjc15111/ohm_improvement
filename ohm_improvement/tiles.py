"""Differentiable programming and drift around a stock AIHWKIT inference tile."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.simulator.tiles.inference import InferenceTile

from ._backend import AnalogTileHolder
from .compensation import convert_compensation
from .conductance import (
    convert_to_conductances,
    convert_back_to_weights,
    _reconstruct_weights,
    ConductanceMapping
)
from .noise import convert_noise_model


@dataclass
class _ProgrammedState:
    realization_id: str
    conductances: list[list[Tensor]]
    coefficients: list[list[Tensor]]
    tensor_versions: tuple[tuple[str, int, int], ...]
    reference: Tensor | None = None


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


class OITile(nn.Module):
    """Differentiable tile for matrix inputs, including unfolded Conv2d.

    Move the stock tile to its target device and dtype before conversion.
    Reprogram after loading a checkpoint, replacing parameters, or updating them.
    Calibration is retained per programmed/drifted state, including its graph.
    """

    supports_indexed = False

    def __init__(
        self,
        tile: InferenceTile,
        g_converter: Optional[BaseConductanceConverter] = None,
        train_significance: bool = True,
        *,
        seed: int,
    ):
        super().__init__()

        _validate_seed(seed)
        noise_model, compensation, converter, mapping = self._prepare_conversion(tile, g_converter)
        self.train(tile.training)
        self.original_drift_compensation = deepcopy(tile.rpu_config.drift_compensation)
        self.compensation = compensation
        self.noise_model = noise_model
        self.g_converter = converter

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
        self._init_conductances(mapping, train_significance)
        self.register_buffer("noise_seed", torch.tensor(seed, dtype=torch.int64, device=self.device))
        self._probe = None
        self._reset_programmed_state()
        # Mutate the backend only after conversion and validation have succeeded.
        tile.get_analog_ctx().set_indexed(False)
        tile.rpu_config.drift_compensation = None
        self.analog_tile_holder.bind(convert_back_to_weights(mapping))

    @staticmethod
    def _prepare_conversion(tile, g_converter=None):
        """Validate and prepare conversion without changing the source tile."""
        if type(tile) is not InferenceTile:
            raise TypeError("OITile only supports conversion of InferenceTile")
        if tile.analog_bias:
            raise ValueError("OITile does not support analog bias; use digital bias")
        if tile.shared_weights is None:
            raise ValueError("OITile requires a tile with shared_weights")
        if tile.in_trans or tile.out_trans:
            raise ValueError("OITile does not support transposed inputs or outputs")
        noise = convert_noise_model(getattr(tile.rpu_config, "noise_model", None))
        if noise is None:
            raise ValueError("OITile requires a PCM noise model")
        compensation = convert_compensation(tile.rpu_config.drift_compensation)
        converter = deepcopy(g_converter if g_converter is not None else noise.g_converter)
        mapping = convert_to_conductances(converter, AnalogTileHolder(tile).weights())
        return noise, compensation, converter, mapping

    def _reset_programmed_state(self) -> None:
        self._programmed: _ProgrammedState | None = None
        self.active_weights: Tensor | None = None
        self._compensation_factor: Tensor | None = None

    def _load_from_state_dict(self, *args, **kwargs):
        self._reset_programmed_state()
        return super()._load_from_state_dict(*args, **kwargs)

    def _check_apply(self, fn) -> None:
        current = self.g_plus[0]
        target = fn(current.new_empty(0))
        if target.device != current.device or target.dtype != current.dtype:
            raise RuntimeError(
                "Move the analog model to the desired device and dtype before OI conversion"
            )

    def _apply(self, fn, recurse=True):
        self._check_apply(fn)
        return super()._apply(fn, recurse)

    def get_analog_tile(self) -> InferenceTile:
        """Return the underlying AIHWKIT analog tile."""
        return self.analog_tile_holder.analog_tile

    @property
    def is_cuda(self) -> bool:
        return self.get_analog_tile().is_cuda

    @property
    def device(self) -> torch.device:
        return self.get_analog_tile().device

    @property
    def in_size(self) -> int:
        return self.get_analog_tile().in_size

    @property
    def out_size(self) -> int:
        return self.get_analog_tile().out_size

    def get_tensor_view(self, ndim: int, dim: Optional[int] = None) -> tuple:
        return self.get_analog_tile().get_tensor_view(ndim, dim)

    def is_indexed(self) -> bool:
        return False

    def set_indexed(self, *args, **kwargs) -> None:
        raise NotImplementedError("OITile does not support indexing; use unfolded Conv2d")

    def _init_conductances(self, mapping: ConductanceMapping, train_significance: bool) -> None:
        self.g_plus = nn.ParameterList(
            [nn.Parameter(g.detach().clone()) for g in mapping.g_plus]
        )
        self.g_minus = nn.ParameterList(
            [nn.Parameter(g.detach().clone()) for g in mapping.g_minus]
        )
        self.significance = nn.Parameter(
            mapping.significance.detach().clone(), requires_grad=train_significance,
        )
        self.register_buffer("scale", mapping.scale.detach().clone())

    def _tensor_versions(self) -> tuple[tuple[str, int, int], ...]:
        return tuple((name, id(value), value._version) for name, value in
                     [*self.named_parameters(), *self.named_buffers()])

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

    def _require_programmed(self) -> None:
        if self._programmed is None:
            raise RuntimeError("Call program() first")
        if self._tensor_versions() != self._programmed.tensor_versions:
            raise RuntimeError("Parameters changed; call program() again")

    def _apply_out_scaling(self, values: Tensor, tensor_view: tuple[int, ...]) -> Tensor:
        if self.out_scaling_alpha is not None:
            return values * self.out_scaling_alpha.view(*tensor_view)
        return values

    def _analog_forward(self, x_input: Tensor) -> Tensor:
        self._require_programmed()
        return self.analog_tile_holder.forward(
            x_input, self.active_weights, is_test=not self.training,
        )

    def _calibration_readout(self) -> Tensor:
        if self._probe is None:
            self._probe = self.compensation.get_readout_tensor(self.in_size).to(self.active_weights)
        output = self.analog_tile_holder.forward(self._probe, self.active_weights, is_test=True)
        return self.compensation.readout(output)

    def _install_conductances(self, gp: list[Tensor], gm: list[Tensor]) -> None:
        self.active_weights = _reconstruct_weights(
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
        self._reset_programmed_state()
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
        self._install_conductances(*programmed)
        reference = self._calibration_readout() if self.compensation is not None else None
        self._programmed = _ProgrammedState(
            realization_id, programmed, coefficients, self._tensor_versions(), reference,
        )

    def drift(self, t_inference: float) -> None:
        """Apply drift/read noise from the retained state, without compounding."""
        self._require_programmed()
        t_inference = float(t_inference)
        if not math.isfinite(t_inference) or t_inference < 0:
            raise ValueError("t_inference must be finite and nonnegative")
        state = self._programmed
        programmed, coefficients = state.conductances, state.coefficients
        drifted = []
        for side, gs, nus in zip(("plus", "minus"), programmed, coefficients):
            drifted.append([
                self.noise_model.apply_drift_noise_to_conductance(
                    g, nu, t_inference,
                    self.noise_sample("read", pair, side, state.realization_id, t_inference))
                for pair, (g, nu) in enumerate(zip(gs, nus))
            ])
        self._install_conductances(*drifted)
        self._compensation_factor = None
        if self.compensation is not None:
            self._compensation_factor = state.reference / self._calibration_readout()

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple[int, ...]] = None
    ) -> Tensor:

        output = self._analog_forward(x_input)
        if self._compensation_factor is not None:
            output = output * self._compensation_factor

        if tensor_view is None:
            tensor_view = self.get_analog_tile().get_tensor_view(output.dim())
        out = self._apply_out_scaling(output, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    @torch.no_grad()
    def clamp_conductances(self) -> None:
        for side in (self.g_plus, self.g_minus):
            for g in side:
                g.clamp_(self.g_converter.g_min, self.g_converter.g_max)

    @torch.no_grad()
    def clamp_significance(self) -> None:
        self.significance.clamp_(0.1, 5.0)

    def to_analog(self) -> InferenceTile:
        """Export independent weights, digital parameters, and cached calibration."""
        self._require_programmed()
        tile = self.analog_tile_holder.snapshot(
            self.active_weights, self._compensation_factor, self._programmed.reference,
        )

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
