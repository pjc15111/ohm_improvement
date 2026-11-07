"""AIHWKIT layout, shared-weight binding, and autograd integration."""

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import Tensor
from torch import nn
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.tiles.inference import InferenceTile


class _ConductanceAnalogFunction(AnalogFunction):
    """Undo AIHWKIT's optimizer normalization for conductance gradients.

    Forward and input gradients retain stock hardware behavior. AIHWKIT's
    weight update divides by mapping scales and its learning-rate scale,
    whereas the derivative of forward multiplies by mapping scales.
    """

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        gradients = list(AnalogFunction.backward(ctx, grad_output))
        tile = ctx.analog_tile
        scales = tile.get_mapping_scales()
        if gradients[3] is not None and scales is not None:
            correction = scales.square() * tile.mapping_lr_scale
            # Native weights are [out, in] on CPU and [in, out] on CUDA.
            correction = correction.view(1, -1) if tile.is_cuda else correction.view(-1, 1)
            gradients[3] = gradients[3] * correction
        return tuple(gradients)


@dataclass
class AnalogTileHolder:
    """Hold the backend without registering it as a PyTorch submodule."""

    analog_tile: InferenceTile

    def weights(self) -> Tensor:
        tile = self.analog_tile
        weights, _ = tile.get_weights(apply_weight_scaling=False)
        weights = weights.detach().to(tile.shared_weights)
        return (weights.T if tile.is_cuda else weights).contiguous().clone()

    def bind(self, weights: Tensor) -> None:
        tile = self.analog_tile
        # First CUDA binding may copy backend weights into this disposable holder.
        tile._parameters.pop("shared_weights", None)
        tile.shared_weights = weights.detach().clone()
        tile.ensure_shared_weights()

    def forward(self, inputs: Tensor, weights: Tensor, *, is_test: bool) -> Tensor:
        tile = self.analog_tile
        return _ConductanceAnalogFunction.apply(
            tile.get_analog_ctx(), tile, inputs, weights, is_test,
        )

    @torch.no_grad()
    def snapshot(
        self, weights: Tensor, alpha: Tensor | None, reference: Tensor | None,
    ) -> InferenceTile:
        """Copy backend storage and calibration without another noisy readout."""
        tile = deepcopy(self.analog_tile)
        tile.ensure_shared_weights()
        tile.shared_weights = nn.Parameter(weights.detach().clone(), requires_grad=False)
        tile.alpha = torch.ones_like(tile.alpha) if alpha is None else alpha.detach().clone()
        if reference is not None:
            tile.drift_baseline = reference.detach().clone()
        tile.ensure_shared_weights(tile.shared_weights)
        return tile
