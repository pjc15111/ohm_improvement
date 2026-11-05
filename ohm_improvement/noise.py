from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

from aihwkit.inference import PCMLikeNoiseModel

import numpy as np
import torch
from torch import Tensor

from typing import Optional, List
from copy import deepcopy


class DifferentiablePCMLikeNoiseModel:
    r"""PCM programming, drift, and read-noise model
    with autograd support.

    Adapted from AIHWKIT's PCMLikeNoiseModel.

    Changes from the original:
        1. Remove @no_grad from all methods.
        2. Accept noise tensors as arguments instead of sampling
           internally, allowing the use of common random numbers.
        3. Avoid in-place arithmetic in the programming-noise
           calculation.
        4. Clamp before the fractional power in the read-noise
           calculation to prevent invalid gradients at zero
           conductance.
        5. We no longer extend BaseNoiseModel.

    The statistical equations and default parameters otherwise follow
    the original model.

    Timing parameters are ordinary scalars. Gradients are supported
    with respect to conductances and drift coefficients, not with
    respect to time.

    Most changes from the original model are documented with 'CHANGE:'.
    Small changes that do not affect semantics (e.g. translating
    `torch_abs` to `torch.abs`) are not documented. All comments aside
    from this block and those marked with CHANGE: are from the
    original.

    Known documentation defects inherited from AIHWKIT'PCMLikeNoiseModel:
    - The programming-noise polynomial specifies the standard deviation,
      not the variance: its value multiplies a standard-normal sample.
    - The polynomial is quadratic only for the default coefficients;
      custom coefficients can specify a different degree.
    - The constructor documentation refers to "prog_coeffs", but the
      argument is named "prog_coeff".

    Known defect inherited from AIHWKIT's PCMLikeNoiseModel:
    The value of the argument prog_coeff_g_max_reference is never used.
    When both prog_coeff and prog_coeff_g_max_reference are not None,
    self.prog_coeff_g_max_reference is not initialized. Subsequently
    computing programming noise raises AttributeError. Retained here
    to keep changes limited to differentiability.

    Original documentation follows.

    Args:
        prog_coeff: Programming polynomial coeffs in
            :math:`\sum_i c_i \left(\frac{g_t}{g_\max}\right)^i`
        g_converter: instantiated class of the conductance converter
            (defaults to single pair)
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        t_read: Parameter of the 1/f fit (in seconds).
        t_0: Parameter of the drift fit (first reading time).

            Note:
                The ``t_inference`` is relative to this time `t0`
                e.g. t_inference counts from the completion of the programming
                of a device.
        prog_noise_scale: Scale for the programming noise.
        read_noise_scale: Scale for the read and accumulated noise.
        drift_scale: Scale for the  drift coefficient.
        prog_coeff_g_max_reference: reference :math:`g_\max` value
            when fitting the coefficients, since the result of the
            polynomial fit is given in uS. If
            ``prog_coeff_g_max_reference`` is not given and
            `prog_coeffs` are given explicitly, it will be set to
            ``g_max`` of the conductance converter.

    """

    # CHANGE: Define the original numerical floor locally rather than
    # importing AIHWKIT's private module-level _ZERO_CLIP constant.
    # Its value is unchanged.
    _ZERO_CLIP = 1e-7

    def __init__(
        self,
        prog_coeff: Optional[List[float]] = None,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = None,
        t_read: float = 250.0e-9,
        t_0: float = 20.0,
        prog_noise_scale: float = 1.0,
        read_noise_scale: float = 1.0,
        drift_scale: float = 1.0,
        prog_coeff_g_max_reference: Optional[float] = None,
    ):
        # CHANGE: Store the converter directly instead of relying on
        # BaseNoiseModel.__init__.
        self.g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)

        self.g_max = getattr(self.g_converter, "g_max", g_max)
        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")

        if prog_coeff_g_max_reference is None:
            self.prog_coeff_g_max_reference = self.g_max

        if prog_coeff is None:
            # standard g_max are defined in respect to 25.0 uS. Need to
            # adjust for that in case g_max is not equal to 25.0 uS
            self.prog_coeff = [0.26348, 1.9650, -1.1731]
            self.prog_coeff_g_max_reference = 25.0
        else:
            self.prog_coeff = prog_coeff

        self.t_0 = t_0
        self.t_read = t_read
        self.prog_noise_scale = prog_noise_scale
        self.read_noise_scale = read_noise_scale
        self.drift_scale = drift_scale

    # CHANGE: No @no_grad decorator
    def apply_programming_noise_to_conductance(
        self,
        g_target: Tensor,
        randn: Tensor,
    ) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 2-degree
        polynomial.
        """
        # CHANGE: The original samples randn_like(g_target) internally.
        # We instead receive randn as an argument, allowing reuse of
        # random samples.

        mat = 1
        sig_prog = self.prog_coeff[0]
        for coeff in self.prog_coeff[1:]:
            # CHANGE: Use out-of-place operations instead of *= and +=.
            # Mutating values can invalidate values saved by autograd
            # for the backward pass.
            mat = mat * (g_target / self.g_max)
            sig_prog = sig_prog + mat * coeff

        # CHANGE: Out-of-place scaling for the same reason.
        sig_prog = sig_prog * (
            self.g_max / self.prog_coeff_g_max_reference
        )  # type: ignore
        g_prog = g_target + self.prog_noise_scale * sig_prog * randn

        # CHANGE: Use clamp rather than clamp_ to avoid mutating an
        # intermediate tensor.
        return g_prog.clamp(min=0.0)  # no negative conductances allowed

    # CHANGE: Remove @no_grad decorator
    def generate_drift_coefficients(
        self,
        g_target: Tensor,
        randn: Tensor
    ) -> Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        # CHANGE: randn is supplied by the caller rather than generated
        # internally to allow reuse of random numbers.

        g_relative = torch.clamp(
            torch.abs(g_target / self.g_max), min=self._ZERO_CLIP
        )

        # gt should be normalized wrt g_max
        mu_drift = (-0.0155 * torch.log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        sig_drift = (-0.0125 * torch.log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        nu_drift = torch.abs(mu_drift + sig_drift * randn).clamp(min=0.0)

        return nu_drift * self.drift_scale

    # CHANGE: No @no_grad decorator
    def apply_drift_noise_to_conductance(
        self,
        g_prog: Tensor,
        drift_noise_param: Tensor,
        t_inference: float,
        randn: Tensor
    ) -> Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        # CHANGE: randn is supplied by the caller rather than generated
        # internally to allow reuse of random numbers.
        t = t_inference + self.t_0

        # drift
        if t > self.t_0:
            g_drift = g_prog * ((t / self.t_0) ** (-drift_noise_param))
        else:
            g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            # CHANGE: Clamp BEFORE the fractional power.
            #
            # Original:
            #   relative.pow(0.65).clamp(min=1e-3)
            #
            # At relative == 0, the power has an infinite derivative.
            # A later clamp can produce 0 * inf during backward,
            # resulting in NaN.
            #
            # For nonnegative x:
            #   max(x**0.65, 1e-3)
            #     == max(x, 1e-3**(1/0.65))**0.65
            #
            # This preserves forward values, up to rounding, while
            # keeping the power's derivative finite.
            relative = (g_prog.abs() / self.g_max).clamp(
                min=1e-3 ** (1.0 / 0.65)
            )
            q_s = (0.0088 / relative.pow(0.65)).clamp(max=0.2)
            sig_noise = q_s * np.sqrt(np.log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch.abs(g_drift) * self.read_noise_scale * sig_noise * randn
        else:
            g_final = g_prog

        return g_final.clamp(min=0.0)


def convert_noise_model(noise_model):
    if noise_model is None:
        return None

    noise_model = deepcopy(noise_model)

    if type(noise_model) is PCMLikeNoiseModel:
        return DifferentiablePCMLikeNoiseModel(
            noise_model.prog_coeff,
            noise_model.g_converter,
            noise_model.g_max,
            noise_model.t_read,
            noise_model.t_0,
            noise_model.prog_noise_scale,
            noise_model.read_noise_scale,
            noise_model.drift_scale,
            noise_model.prog_coeff_g_max_reference
        )

    raise TypeError(
        f"Unsupported drift compensation: {type(noise_model).__name__}"
    )
