from aihwkit.simulator.tiles.functions import AnalogFunction

from aihwkit.inference.compensation.drift import GlobalDriftCompensation

import torch
from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery


class DifferentiableBaseDriftCompensation:
    r"""Base class for drift compensations with autograd support.

    Adapted from AIHWKIT's BaseDriftCompensation.

    Changes from the original:
        1. Remove @no_grad from all methods.
        2. Modify `init_baseline` to generate the probe and manage the
           AnalogFunction call itself to avoid calling in to code that
           does not support autograd.
        3. Change interface of `apply` to actually apply compensation,
           allowing easier implementation of affine compensation.

    Changes from the original model are documented with 'CHANGE:'.
    All comments aside from this block and those marked with CHANGE:
    are from the original.

    Original documentation follows.

    Base class for drift compensations.
    """

    def __init__(self) -> None:
        pass

    # CHANGE: No @no_grad decorator
    # CHANGE: Correct the original return annotation; this method
    # returns a single reference Tensor.
    # CHANGE: Now accepts the weights as an argument to let us manage
    # lifetime manually.
    def init_baseline(
        self,
        tile: "InferenceTileWithPeriphery",
        weights: Tensor
    ) -> Tensor:
        """Initialize the base line for applying the compensation.

        Uses a all one tensor for read_out.

        Args:
            tile: forward output of the read out vector to compensate

        Returns:
            reference tensor readout
        """
        # CHANGE: We no longer call tile._forward_drift_readout_tensor
        # as this function is not differentiable. Instead we get the
        # readout tensor ourselves and directly apply the tile's
        # analog function.
        probe = self.get_readout_tensor(tile.in_size).to(tile.shared_weights)
        forward_output = AnalogFunction.apply(
            tile.get_analog_ctx(),
            tile,
            probe,
            weights,
            True,
        )

        ref_value = self.readout(forward_output)

        return ref_value

    # CHANGE: No @no_grad decorator
    def get_readout_tensor(self, in_size: int) -> Tensor:
        """Return the read-out tensor.

        Called once during :meth:`~init_baseline`.
        """
        raise NotImplementedError

    # CHANGE: No @no_grad decorator
    def readout(self, out_tensor: Tensor) -> Tensor:
        """Implement the read out math."""
        raise NotImplementedError

    # CHANGE: No @no_grad decorator
    # CHANGE: This function now accepts an additional argument,
    # `output`. This is the output to which compensation should be
    # applied. Compensation is then applied to this output by this
    # function rather than by the caller.
    def apply(
        self,
        forward_output: Tensor,
        ref_value: Tensor,
        output: Tensor
    ) -> Tensor:
        """CHANGE: Calculate drift compensation from the calibration
        readout and apply it to output."""
        current_value = self.readout(forward_output)
        ratio = ref_value / current_value

        return output * ratio


class DifferentiableGlobalDriftCompensation(DifferentiableBaseDriftCompensation):
    r"""Global linear drift compensation with autograd support.

    Adapted from AIHWKIT's GlobalDriftCompensation.

    Changes from the original:
        1. Remove @no_grad from all methods.

    Most changes from the original model are documented with 'CHANGE:'.
    Small changes that do not affect semantics (e.g. translating
    `torch_abs` to `torch.abs`) are not documented. All comments aside
    from this block and those marked with CHANGE: are from the
    original.

    Original documentation follows.

    Uses a constant factor for compensating the drift.
    """

    # CHANGE: No @no_grad decorator
    def readout(self, out_tensor: Tensor) -> Tensor:
        """Read outs the mean abs."""
        return torch.clamp(torch.abs(out_tensor).mean(), min=0.0001)

    # CHANGE: No @no_grad decorator
    def get_readout_tensor(self, in_size: int) -> Tensor:
        """Return the read-out tensor.

        Uses the set of one-hot vectors (eye).
        """
        return torch.eye(in_size)

    def __str__(self) -> str:
        return "{}()".format(self.__class__.__name__)


def convert_compensation(compensation):
    if compensation is None:
        return None

    if type(compensation) is GlobalDriftCompensation:
        return DifferentiableGlobalDriftCompensation()

    raise TypeError(
        f"Unsupported drift compensation: {type(compensation).__name__}"
    )
