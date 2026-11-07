"""Differentiable drift compensation using AIHWKIT's readout equations."""

import torch
from torch import Tensor
from aihwkit.inference.compensation.drift import GlobalDriftCompensation


class DifferentiableBaseDriftCompensation:
    """Compute calibration readouts and apply a differentiable scale factor."""

    def get_readout_tensor(self, in_size: int) -> Tensor:
        raise NotImplementedError

    def readout(self, out_tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def apply(self, forward_output: Tensor, ref_value: Tensor, output: Tensor) -> Tensor:
        return output * (ref_value / self.readout(forward_output))


class DifferentiableGlobalDriftCompensation(DifferentiableBaseDriftCompensation):
    """Use the mean absolute response to one-hot inputs for global scaling."""

    def readout(self, out_tensor: Tensor) -> Tensor:
        return out_tensor.abs().mean().clamp(min=0.0001)

    def get_readout_tensor(self, in_size: int) -> Tensor:
        return torch.eye(in_size)


def convert_compensation(compensation) -> DifferentiableBaseDriftCompensation | None:
    if compensation is None:
        return None
    if type(compensation) is GlobalDriftCompensation:
        return DifferentiableGlobalDriftCompensation()
    raise TypeError(f"Unsupported drift compensation: {type(compensation).__name__}")
