import pytest
import torch
import aihwkit.inference.noise.pcm as stock_pcm

from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
)

from ohm_improvement.noise import (
    DifferentiablePCMLikeNoiseModel,
    convert_noise_model,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def test_conversion_preserves_configuration():
    stock = stock_pcm.PCMLikeNoiseModel(
        g_converter=SinglePairConductanceConverter(g_min=1.0, g_max=30.0),
        prog_coeff=[0.2, 0.7, 0.1],
        t_read=1e-6,
        t_0=15.0,
        prog_noise_scale=0.4,
        read_noise_scale=0.6,
        drift_scale=0.8,
    )
    # Set directly to avoid the stock constructor's known reference bug.
    stock.prog_coeff_g_max_reference = 27.0

    actual = convert_noise_model(stock)

    for name in (
        "g_max",
        "t_read",
        "t_0",
        "prog_noise_scale",
        "read_noise_scale",
        "drift_scale",
        "prog_coeff_g_max_reference",
    ):
        assert getattr(actual, name) == getattr(stock, name)

    assert actual.prog_coeff == stock.prog_coeff
    assert actual.g_converter.g_min == stock.g_converter.g_min
    assert actual.g_converter.g_max == stock.g_converter.g_max

    # Conversion must not leave mutable configuration shared with stock.
    actual.prog_coeff[0] = 999.0
    actual.g_converter.g_max = 999.0
    assert stock.prog_coeff[0] == 0.2
    assert stock.g_converter.g_max == 30.0


@pytest.mark.parametrize("time", [0.0, 86400.0, 31536000.0])
def test_noise_values_match_stock_with_identical_samples(
    monkeypatch, device, time
):
    stock = stock_pcm.PCMLikeNoiseModel()
    actual = convert_noise_model(stock)

    targets = torch.tensor(
        [0.0, 1e-8, 0.01, 1.0, 10.0, 25.0],
        device=device,
        dtype=torch.float64,
    )
    sample = targets.new_tensor([-0.8, 0.3, -0.2, 1.1, -1.3, 0.6])

    # Override only the random sampler used by the stock PCM module.
    monkeypatch.setattr(
        stock_pcm, "randn_like", lambda tensor: sample.to(tensor).clone()
    )

    expected_programmed = stock.apply_programming_noise_to_conductance(
        targets
    )
    programmed = actual.apply_programming_noise_to_conductance(
        targets, sample
    )
    torch.testing.assert_close(programmed, expected_programmed)

    expected_nu = stock.generate_drift_coefficients(targets)
    nu = actual.generate_drift_coefficients(targets, sample)
    torch.testing.assert_close(nu, expected_nu)

    expected = stock.apply_drift_noise_to_conductance(
        expected_programmed, expected_nu, time
    )
    result = actual.apply_drift_noise_to_conductance(
        programmed, nu, time, sample
    )
    torch.testing.assert_close(result, expected)


def test_zero_conductance_has_finite_gradients(device):
    noise = DifferentiablePCMLikeNoiseModel()
    targets = torch.tensor(
        [0.0, 1e-10, 0.1, 5.0, 20.0],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    programmed = noise.apply_programming_noise_to_conductance(
        targets, torch.zeros_like(targets)
    )
    nu = noise.generate_drift_coefficients(
        targets, torch.full_like(targets, 0.2)
    )
    result = noise.apply_drift_noise_to_conductance(
        programmed, nu, 86400.0, torch.full_like(targets, 0.3)
    )
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert targets.grad is not None
    assert torch.isfinite(targets.grad).all()
    assert targets.grad[2:].abs().sum() > 0


def test_explicit_programming_reference_is_used():
    noise = DifferentiablePCMLikeNoiseModel(
        g_max=30.0,
        prog_coeff=[0.2, 0.4],
        prog_coeff_g_max_reference=20.0,
    )
    g = torch.tensor([1.0, 10.0])
    expected = g + (0.2 + 0.4 * g / 30.0) * (30.0 / 20.0)

    actual = noise.apply_programming_noise_to_conductance(
        g, torch.ones_like(g)
    )
    torch.testing.assert_close(actual, expected)
