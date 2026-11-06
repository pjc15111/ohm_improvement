import pytest
import torch

from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
    DualPairConductanceConverter,
    NPairConductanceConverter,
)

from ohm_improvement.conductance import (
    ConductanceMapping,
    convert_to_conductances,
    convert_back_to_weights,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize(
    "converter",
    [
        SinglePairConductanceConverter(g_min=0.0, g_max=25.0),
        SinglePairConductanceConverter(g_min=1.0, g_max=25.0),
        DualPairConductanceConverter(
            f_lst=[1.0, 2.0], g_min=0.0, g_max=25.0
        ),
        NPairConductanceConverter(
            f_lst=[1.0, 2.0, 4.0], g_min=1.0, g_max=25.0
        ),
    ],
)
@pytest.mark.parametrize("all_zero", [False, True])
def test_conversion_matches_stock_and_round_trips(
    device, converter, all_zero
):
    weights = torch.tensor(
        [[-1.0, -0.17, 0.0], [0.03, 0.42, 0.91]],
        device=device,
        dtype=torch.float64,
    )
    if all_zero:
        weights.zero_()

    mapping = convert_to_conductances(converter, weights)
    stock_conductances, stock_params = converter.convert_to_conductances(
        weights
    )

    assert len(mapping.g_plus) == len(stock_conductances) // 2
    assert len(mapping.g_minus) == len(mapping.g_plus)

    for actual, expected in zip(mapping.g_plus, stock_conductances[::2]):
        torch.testing.assert_close(actual, expected)

    for actual, expected in zip(mapping.g_minus, stock_conductances[1::2]):
        torch.testing.assert_close(actual, expected)

    expected_factors = getattr(converter, "f_lst", [1.0])
    torch.testing.assert_close(
        mapping.significance,
        weights.new_tensor(expected_factors),
    )
    torch.testing.assert_close(mapping.scale, stock_params["scale_ratio"])
    torch.testing.assert_close(convert_back_to_weights(mapping), weights)

    for tensor in [
        *mapping.g_plus,
        *mapping.g_minus,
        mapping.significance,
        mapping.scale,
    ]:
        assert tensor.device == weights.device
        assert tensor.dtype == weights.dtype
        assert torch.isfinite(tensor).all()


def test_reconstruction_gradients():
    # Distinct, interior values avoid clamp boundaries and ensure every
    # pair contributes. gradcheck compares autograd with finite differences.
    gp0 = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    gp1 = torch.tensor([[5.0, 7.0]], dtype=torch.float64, requires_grad=True)
    gm0 = torch.tensor([[0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    gm1 = torch.tensor([[1.0, 1.5]], dtype=torch.float64, requires_grad=True)
    factors = torch.tensor([1.2, 2.3], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)

    def reconstruct(a, b, c, d, f, s):
        return convert_back_to_weights(
            ConductanceMapping([a, b], [c, d], f, s)
        )

    assert torch.autograd.gradcheck(
        reconstruct, (gp0, gp1, gm0, gm1, factors, scale)
    )


@pytest.mark.parametrize("case", ["empty", "missing_minus", "wrong_factors"])
def test_reconstruction_rejects_mismatched_pairs(case):
    g = torch.ones(2, 3)

    if case == "empty":
        mapping = ConductanceMapping([], [], torch.empty(0), torch.tensor(1.0))
    elif case == "missing_minus":
        mapping = ConductanceMapping(
            [g, g], [g], torch.ones(2), torch.tensor(1.0)
        )
    else:
        mapping = ConductanceMapping(
            [g], [g], torch.ones(2), torch.tensor(1.0)
        )

    with pytest.raises(ValueError):
        convert_back_to_weights(mapping)


def test_unknown_subclass_is_not_silently_accepted():
    class UnsupportedConverter(SinglePairConductanceConverter):
        pass

    with pytest.raises(NotImplementedError):
        convert_to_conductances(UnsupportedConverter(), torch.ones(2, 3))
