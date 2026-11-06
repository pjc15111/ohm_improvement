"""Evaluate programming schemes using stock AIHWKIT conductance noise.

Conductances use canonical [out_features, in_features] layout. Tile names match
those in the original analog model, retained before conversion to OI.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from aihwkit.inference.converter.base import BaseConductanceConverter

from aihwkit.simulator.tiles.inference import InferenceTile

from ohm_improvement.conductance import (
    ConductanceMapping, convert_to_conductances, convert_back_to_weights,
)
from ohm_improvement.model import AnalogModel, OIModel


@dataclass
class TileProgrammingScheme:
    """Detached target conductances and digital parameters for one tile.

    ConductanceMapping contains positive/negative pairs, significance, and scale.
    None for a digital parameter means retain the original analog model's value.
    Mapping scales remain those of the original analog model.
    """

    conductances: ConductanceMapping
    output_scaling: Optional[Tensor] = None
    bias: Optional[Tensor] = None

    def detached_copy(
        self, *, device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "TileProgrammingScheme":
        """Copy all tensors into independent storage, without autograd history."""
        def snapshot(t: Tensor) -> Tensor:
            return t.detach().to(device=device, dtype=dtype).clone()

        mapping = self.conductances
        return TileProgrammingScheme(
            conductances=ConductanceMapping(
                g_plus=[snapshot(g) for g in mapping.g_plus],
                g_minus=[snapshot(g) for g in mapping.g_minus],
                significance=snapshot(mapping.significance),
                scale=snapshot(mapping.scale),
            ),
            output_scaling=snapshot(self.output_scaling) if self.output_scaling is not None else None,
            bias=snapshot(self.bias) if self.bias is not None else None,
        )


@dataclass
class ProgrammedTileRealization:
    """One realization, retained unchanged while evaluating different times."""

    g_plus: list[Tensor]
    g_minus: list[Tensor]
    drift_plus: list[Optional[Tensor]]
    drift_minus: list[Optional[Tensor]]
    compensation_baseline: Optional[Tensor]


def extract_oi_programming(oi_model: OIModel) -> dict[str, TileProgrammingScheme]:
    """Snapshot optimized targets and digital parameters without simulating noise."""
    if not isinstance(oi_model, OIModel):
        raise TypeError("Expected OIModel")
    schemes = {}
    for name, tile in oi_model.named_oi_tiles():

        def canonical(g):
            g = g.detach()
            return (g.T if tile.get_analog_tile().is_cuda else g).contiguous().clone()

        schemes[name] = TileProgrammingScheme(
            conductances=ConductanceMapping(
                g_plus=[canonical(g) for g in tile.g_plus],
                g_minus=[canonical(g) for g in tile.g_minus],
                significance=tile.significance.detach().clone(),
                scale=tile.scale.detach().clone(),
            ),
            output_scaling=(tile.out_scaling_alpha.detach().clone()
                            if tile.out_scaling_alpha is not None else None),
            bias=tile.bias.detach().clone() if tile.digital_bias else None,
        )
    return schemes


def extract_analog_programming(
    analog_model: AnalogModel,
    *,
    converter: Optional[BaseConductanceConverter] = None,
) -> dict[str, TileProgrammingScheme]:
    """Snapshot targets using an override converter, or each tile's own converter.

    Pass the converter returned by WPO without changing the model configuration.

    Read nominal tile weights without applying mapping/output scales. Custom
    converters may require CPU initialization before transfer to the tile device.
    """
    if not isinstance(analog_model, AnalogModel):
        raise TypeError("Expected an AIHWKIT analog model or layer")
    schemes = {}
    with torch.no_grad():
        for name, tile in analog_model.named_modules():
            if not isinstance(tile, InferenceTile):
                continue
            weights, _ = tile.get_weights(apply_weight_scaling=False)
            weights = weights.to(device=tile.device)
            tile_converter = (converter if converter is not None
                              else tile.rpu_config.noise_model.g_converter)
            mapping = convert_to_conductances(tile_converter, weights)
            schemes[name] = TileProgrammingScheme(
                conductances=ConductanceMapping(
                    g_plus=[g.detach().clone() for g in mapping.g_plus],
                    g_minus=[g.detach().clone() for g in mapping.g_minus],
                    significance=mapping.significance.detach().clone(),
                    scale=mapping.scale.detach().clone(),
                ),
                output_scaling=(tile.out_scaling_alpha.detach().clone()
                                if tile.out_scaling_alpha is not None else None),
                bias=tile.bias.detach().clone() if tile.digital_bias else None,
            )
    return schemes


class StockProgrammingEvaluator:
    """Install stock-noised conductances into a retained original analog model.

    The supplied model is reused and mutated in place, in evaluation mode.
    Its stock noise models, compensation, mapping scales, and forward settings
    govern evaluation. Neither extraction nor evaluation modifies the OI model.
    Seeds control PyTorch conductance noise; noisy backend forward/calibration
    calls use a separate AIHWKIT RNG and need not repeat exactly.
    """

    _model: AnalogModel
    _tiles: list[tuple[str, InferenceTile, TileProgrammingScheme]]
    _current_programmed: dict[str, ProgrammedTileRealization]
    _seed: int
    _realization_id: Optional[str]

    def __init__(
        self,
        analog_model: AnalogModel,
        schemes: Mapping[str, TileProgrammingScheme],
        *,
        seed: int,
    ) -> None:
        """Validate tile correspondence and retain a detached programming snapshot.

        Pair each tile with its name and a detached scheme snapshot; retain only the
        current realization in _current_programmed (one entry per tile). Store
        one base seed; timestep seeds are derived on demand.

        The seed selects an independent evaluation stream; stock RNG calls must
        be isolated from the caller's RNG state. Do not program weights yet.
        """
        if not isinstance(analog_model, AnalogModel):
            raise TypeError("Expected an AIHWKIT analog model or layer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        tiles = {name: tile for name, tile in analog_model.named_modules()
                 if isinstance(tile, InferenceTile)}
        if not tiles or tiles.keys() != schemes.keys():
            raise ValueError("schemes must match the analog model's inference tiles")
        self._tiles = []
        for name, tile in tiles.items():
            if tile.analog_bias:
                raise ValueError("analog bias columns are not supported")
            scheme = schemes[name].detached_copy(device=tile.device, dtype=tile.get_dtype())
            mapping = scheme.conductances
            weights = convert_back_to_weights(mapping)
            if (weights.shape != (tile.out_size, tile.in_size)
                    or any(g.shape != weights.shape for g in mapping.g_plus + mapping.g_minus)
                    or not all(torch.isfinite(t).all() for t in [
                        *mapping.g_plus, *mapping.g_minus, mapping.significance, mapping.scale])
                    or mapping.scale.numel() != 1 or mapping.scale.item() <= 0):
                raise ValueError(f"Invalid conductance mapping for tile {name!r}")
            if scheme.bias is not None and not tile.digital_bias:
                raise ValueError(f"Tile {name!r} has no digital bias")
            if scheme.output_scaling is not None and tile.out_scaling_alpha is None:
                raise ValueError(f"Tile {name!r} has no output scaling parameter")
            for source, target in [(scheme.bias, tile.bias if tile.digital_bias else None),
                                   (scheme.output_scaling, tile.out_scaling_alpha)]:
                if source is not None and source.shape != target.shape:
                    raise ValueError(f"Digital parameter shape mismatch for tile {name!r}")
            self._tiles.append((name, tile, scheme))
        self._model = analog_model.eval()
        self._seed = seed
        self._realization_id = None
        self._current_programmed = {}

    @property
    def model(self) -> AnalogModel:
        """The supplied analog model, ready for inference after program/drift."""
        return self._model

    def program(self, realization_id: str) -> None:
        """Sample stock programming noise and drift coefficients once per pair.

        realization_id is a meaningful name such as "train1", "validation1", or
        "test1". The name selects a reproducible noise realization from one base
        seed; no separate train/validation/evaluation seeds are needed.

        Install programmed weights and initialize stock compensation baselines.
        Replace _current_programmed for all tiles, discarding the previous
        realization; do not cache earlier realizations. Calling program again,
        even with the same ID, rebuilds the state from the target schemes.
        Reset compensation factors. After this call, the model represents
        programming completion, before drift/read noise.
        """
        if not isinstance(realization_id, str):
            raise TypeError("realization_id must be a string")
        self._current_programmed = {}
        self._realization_id = None
        for name, tile, scheme in self._tiles:
            self._current_programmed[name] = self.program_one(name, tile, scheme, realization_id)
        self._realization_id = realization_id

    @torch.no_grad()
    def program_one(
        self, name: str, tile: InferenceTile, scheme: TileProgrammingScheme,
        realization_id: str,
    ) -> ProgrammedTileRealization:
        """Program one tile, install its weights, and return its realization."""
        if scheme.bias is not None:
            tile.bias.copy_(scheme.bias)
        if scheme.output_scaling is not None:
            tile.out_scaling_alpha.copy_(scheme.output_scaling)
        noise = tile.rpu_config.noise_model
        programmed, coefficients = [], []
        for side, targets in (("plus", scheme.conductances.g_plus),
                              ("minus", scheme.conductances.g_minus)):
            gp, nu = [], []
            for pair, target in enumerate(targets):
                with self._rng(tile, self._seed_for(name, realization_id, "program",
                              pair_index=pair, side=side)):
                    gp.append(noise.apply_programming_noise_to_conductance(target))
                with self._rng(tile, self._seed_for(name, realization_id, "drift",
                              pair_index=pair, side=side)):
                    nu.append(noise.generate_drift_coefficients(target))
            programmed.append(gp)
            coefficients.append(nu)
        mapping = scheme.conductances
        weights = convert_back_to_weights(ConductanceMapping(
            *programmed, mapping.significance, mapping.scale))
        self._install_weights(tile, weights)
        tile.alpha = torch.ones_like(tile.alpha)
        # Exact-reference compensation reads the nominal target matrix.
        tile.reference_combined_weights = convert_back_to_weights(
            scheme.conductances).detach().clone()
        compensation = tile.rpu_config.drift_compensation
        baseline = None
        if compensation is not None:
            with self._rng(tile, self._seed_for(name, realization_id, "baseline",
                          pair_index=0, side="calibration")):
                baseline = compensation.init_baseline(tile)
        tile.drift_baseline = baseline
        return ProgrammedTileRealization(
            *programmed, *coefficients, baseline)

    def drift(self, t_inference: float) -> None:
        """Apply stock drift/read noise from the stored programmed conductances.

        Requires program(). Reuse the current programmed conductances and drift
        coefficients without reprogramming or modifying the stored state. Each
        call starts from programming completion, so drift does not compound.
        Times are elapsed seconds since programming; t=0
        still invokes the stock conductance-level drift/read-noise method.
        Derive read-noise seeds from the base seed, tile name, realization,
        pair, side, and elapsed time so calls may be repeated or reordered.
        Install reconstructed weights and update stock compensation factors.
        """
        if self._realization_id is None:
            raise RuntimeError("Call program() before drift()")
        t_inference = float(t_inference)
        if not math.isfinite(t_inference) or t_inference < 0:
            raise ValueError("t_inference must be finite and nonnegative")
        for name, tile, scheme in self._tiles:
            self.drift_one(name, tile, scheme, t_inference)

    @torch.no_grad()
    def drift_one(
        self, name: str, tile: InferenceTile, scheme: TileProgrammingScheme,
        t_inference: float,
    ) -> None:
        """Drift one tile from its current realization; called by drift()."""
        state = self._current_programmed[name]
        noise = tile.rpu_config.noise_model
        drifted = []
        for side, conductances, coefficients in (
            ("plus", state.g_plus, state.drift_plus),
            ("minus", state.g_minus, state.drift_minus),
        ):
            gs = []
            for pair, (g, nu) in enumerate(zip(conductances, coefficients)):
                with self._rng(tile, self._seed_for(name, self._realization_id, "read",
                              pair_index=pair, side=side, t_inference=t_inference)):
                    gs.append(noise.apply_drift_noise_to_conductance(g, nu, t_inference))
            drifted.append(gs)
        mapping = scheme.conductances
        self._install_weights(tile, convert_back_to_weights(ConductanceMapping(
            *drifted, mapping.significance, mapping.scale)))
        compensation = tile.rpu_config.drift_compensation
        if compensation is not None:
            with self._rng(tile, self._seed_for(name, self._realization_id, "readout",
                          pair_index=0, side="calibration", t_inference=t_inference)):
                readout = tile._forward_drift_readout_tensor()
                tile.alpha = compensation.apply(readout, state.compensation_baseline)

    def _install_weights(self, tile: InferenceTile, weights: Tensor) -> None:
        """Install canonical weights without remapping or applying scaling twice.

        Preserve digital bias and scales, and handle CPU/CUDA backend layout and
        shared-weight initialization without overwriting the supplied weights.
        """
        # First binding may copy backend weights into shared storage on CUDA.
        # Complete that before installing the supplied canonical weights.
        tile.ensure_shared_weights()
        tile.set_weights(weights, apply_weight_scaling=False)

    def _seed_for(
        self,
        tile_name: str,
        realization_id: str,
        stage: str,
        *,
        pair_index: int,
        side: str,
        t_inference: Optional[float] = None,
    ) -> int:
        """Derive a stable seed for programming, drift coefficients, or read noise.

        Hash the string realization ID stably, with a canonical elapsed time.
        Programming and drift-coefficient seeds omit time. Calibration has a
        separate stage; backend RNG limitations are documented in _rng().
        """
        timestamp = None if t_inference is None else float(t_inference or 0.).hex()
        key = json.dumps([self._seed, tile_name, realization_id, stage,
                          pair_index, side, timestamp], ensure_ascii=True)
        return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(),
                              "little") % (2**63)


    @staticmethod
    @contextmanager
    def _rng(tile: InferenceTile, seed: int):
        """Isolate stock noise's PyTorch RNG on CPU and the selected GPU.

        AIHWKIT's backend forward/calibration noise uses its own RNG, which is
        not controlled here. Exact repeatability assumes perfect forward I/O.
        """
        devices = [tile.device.index] if tile.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if tile.is_cuda:
                torch.cuda.default_generators[tile.device.index].manual_seed(seed)
            yield


def evaluator_from_oi(
    oi_model: OIModel, analog_model: AnalogModel, *, seed: int,
) -> StockProgrammingEvaluator:
    """Use OI targets with the retained original analog model, mutated in place."""
    return StockProgrammingEvaluator(analog_model, extract_oi_programming(oi_model), seed=seed)


def evaluator_from_analog(
    analog_model: AnalogModel, *, seed: int,
    converter: Optional[BaseConductanceConverter] = None,
) -> StockProgrammingEvaluator:
    """Use stock or WPO targets and evaluate in the supplied analog model."""
    return StockProgrammingEvaluator(
        analog_model, extract_analog_programming(analog_model, converter=converter), seed=seed,
    )


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> dict[str, float | int]:
    """Return mean loss, accuracy (0–1), and count for classification examples.

    Accept logits or log probabilities shaped [batch, classes] and integer class
    targets. Leave the model in eval mode; move batches to device, not the model.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        total_loss += F.cross_entropy(output, targets, reduction="sum").item()
        correct += (output.argmax(dim=-1) == targets).sum().item()
        count += targets.numel()
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return {"loss": total_loss / count, "accuracy": correct / count, "count": count}


def evaluate_programming(
    evaluator: StockProgrammingEvaluator,
    evaluate: Callable[[AnalogModel], Mapping[str, float | int]],
    realization_ids: Iterable[str],
    times: Iterable[float],
) -> list[dict[str, str | float | int]]:
    """Return one metrics record per realization/time, without averaging curves.

    Program once per realization, then drift from that state for each elapsed
    time in seconds. Call evaluate(model) after each drift; its returned metrics
    are recorded with the realization ID and elapsed time. The callback controls
    gradient tracking and can capture its own data/device configuration.
    The model is passed in eval mode and left at the last realization/time.
    """
    times = tuple(times)
    results = []
    for realization_id in realization_ids:
        evaluator.program(realization_id)
        for time in times:
            evaluator.drift(time)
            metrics = evaluate(evaluator.model.eval())
            results.append({
                **metrics,
                "realization_id": realization_id,
                "t_inference": float(time),
            })
    return results
