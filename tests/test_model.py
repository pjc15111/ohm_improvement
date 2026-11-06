import pytest
import torch
from torch import nn

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import NPairConductanceConverter
from aihwkit.simulator.configs import InferenceRPUConfig

from ohm_improvement.model import (
    convert_to_oi,
    oi_to_analog,
    program_weights,
    optimization_parameters,
    clamp_model,
)
from ohm_improvement.tiles import OITile


def make_model(bias=True, scaling=True):
    torch.manual_seed(12)
    digital = nn.Sequential(
        nn.Sequential(nn.Linear(3, 4, bias=bias), nn.Tanh()),
        nn.Linear(4, 2, bias=bias),
    ).eval()

    config = InferenceRPUConfig()
    config.forward.is_perfect = True
    config.mapping.max_input_size = 0
    config.mapping.max_output_size = 0
    config.mapping.learn_out_scaling = scaling
    config.mapping.weight_scaling_omega = 0.7 if scaling else 0.0
    config.drift_compensation = GlobalDriftCompensation()
    config.noise_model.g_converter = NPairConductanceConverter([1.0, 2.0])
    config.noise_model.prog_noise_scale = 0.0
    config.noise_model.drift_scale = 0.0
    config.noise_model.read_noise_scale = 0.0

    analog = convert_to_analog(digital, rpu_config=config).eval()
    return analog


def oi_tiles(model):
    return [module for module in model.modules() if isinstance(module, OITile)]


def test_nested_conversion_and_model_programming():
    original = make_model()
    converted = convert_to_oi(original, 2, [0.0, 86400.0]).eval()

    assert converted is not original
    assert len(oi_tiles(converted)) == 2
    assert not oi_tiles(original)
    assert all(tile.active_weights is None for tile in oi_tiles(converted))

    program_weights(converted, 1, 1)
    assert all(tile.active_weights is not None for tile in oi_tiles(converted))

    x = torch.tensor([[0.2, -0.3, 0.8], [-0.1, 0.7, 0.4]])
    torch.testing.assert_close(
        converted(x), original(x), atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize(
    "option",
    ["include_conductances", "include_output_scaling",
     "include_bias", "include_significance"],
)
def test_parameter_selection_returns_exact_group(option):
    model = convert_to_oi(make_model(), 1, [0.0])
    tiles = oi_tiles(model)
    options = {
        "include_conductances": False,
        "include_output_scaling": False,
        "include_bias": False,
        "include_significance": False,
    }
    options[option] = True

    if option == "include_conductances":
        expected = [
            p for tile in tiles for side in (tile.g_plus, tile.g_minus)
            for p in side
        ]
    elif option == "include_output_scaling":
        expected = [
            tile.out_scaling_alpha for tile in tiles
            if tile.out_scaling_alpha is not None
        ]
    elif option == "include_bias":
        expected = [tile.bias for tile in tiles]
    else:
        expected = [tile.significance for tile in tiles]

    actual = optimization_parameters(model, **options)
    assert len(actual) == 1
    assert actual[0]["name"] == option.removeprefix("include_")
    assert [id(p) for p in actual[0]["params"]] == [id(p) for p in expected]


def test_absent_optional_parameters_are_not_returned():
    model = convert_to_oi(make_model(bias=False, scaling=False), 1, [0.0])

    actual = optimization_parameters(
        model,
        include_conductances=False,
        include_output_scaling=True,
        include_bias=True,
        include_significance=False,
    )
    assert actual == []


def test_model_clamping_reaches_every_tile():
    model = convert_to_oi(make_model(), 1, [0.0])
    with torch.no_grad():
        for tile in oi_tiles(model):
            for g in tile.g_plus:
                g.fill_(-2.0)
            for g in tile.g_minus:
                g.fill_(100.0)
            tile.significance.fill_(9.0)

    clamp_model(model)

    for tile in oi_tiles(model):
        for g in tile.g_plus:
            torch.testing.assert_close(
                g, torch.full_like(g, tile.g_converter.g_min)
            )
        for g in tile.g_minus:
            torch.testing.assert_close(
                g, torch.full_like(g, tile.g_converter.g_max)
            )
        torch.testing.assert_close(
            tile.significance, torch.full_like(tile.significance, 5.0)
        )


def test_export_does_not_copy_graphs_or_mutate_original():
    model = convert_to_oi(make_model(), 1, [0.0]).eval()
    program_weights(model, 0, 0)
    tiles = oi_tiles(model)

    # Export must work with live non-leaf tensors on the OITiles.
    assert all(tile.active_weights.grad_fn is not None for tile in tiles)
    active_ids = [id(tile.active_weights) for tile in tiles]

    x = torch.tensor([[0.2, -0.3, 0.8]])
    expected = model(x).detach().clone()
    exported = oi_to_analog(model)

    assert exported is not model
    assert not oi_tiles(exported)
    assert oi_tiles(model) == tiles
    assert [id(tile.active_weights) for tile in tiles] == active_ids
    torch.testing.assert_close(
        exported(x), expected, atol=3e-6, rtol=3e-5
    )

    # The source graph remains usable after export.
    model(x).square().sum().backward()
    for tile in tiles:
        for g in [*tile.g_plus, *tile.g_minus]:
            assert g.grad is not None
            assert torch.isfinite(g.grad).all()

    # Exported parameters must not alias the source model's parameters.
    with torch.no_grad():
        for parameter in exported.parameters():
            parameter.zero_()

    torch.testing.assert_close(
        model(x), expected, atol=3e-6, rtol=3e-5
    )


def test_inplace_export_preserves_outer_model_identity():
    model = convert_to_oi(make_model(), 1, [0.0]).eval()
    program_weights(model, 0, 0)

    exported = oi_to_analog(model, inplace=True)

    assert exported is model
    assert not oi_tiles(model)


def test_model_noise_seeds_are_reproducible_and_distinct_per_tile():
    seeds = dict(train=11, validation=22, evaluation=33)
    analog = make_model()
    rng = torch.random.get_rng_state().clone()
    first = convert_to_oi(analog, 2, [0., 86400.], noise_seeds=seeds)
    second = convert_to_oi(analog, 2, [0., 86400.], noise_seeds=seeds)
    assert torch.equal(rng, torch.random.get_rng_state())
    tiles = oi_tiles(first)
    assert not torch.equal(tiles[0].noise_seeds, tiles[1].noise_seeds)
    for a, b in zip(tiles, oi_tiles(second)):
        torch.testing.assert_close(a.noise_seeds, b.noise_seeds)
    program_weights(first, 0, 0, noise_split="validation")
    with pytest.raises(ValueError, match="split"):
        program_weights(first, 0, 0, noise_split="typo")


@pytest.mark.parametrize("seeds", [dict(train=1, validation=1, evaluation=2),
                                   dict(train=-1, validation=2, evaluation=3),
                                   dict(train=1)])
def test_invalid_noise_seeds(seeds):
    with pytest.raises(ValueError):
        convert_to_oi(make_model(), 1, [0.], noise_seeds=seeds)


def test_default_noise_seeds_follow_manual_seed():
    analog = make_model()
    torch.manual_seed(123)
    first = convert_to_oi(analog, 1, [0.])
    torch.manual_seed(123)
    second = convert_to_oi(analog, 1, [0.])
    for a, b in zip(oi_tiles(first), oi_tiles(second)):
        torch.testing.assert_close(a.noise_seeds, b.noise_seeds)



@pytest.mark.parametrize("overrides", [{}, dict(conductance_lr=.003, significance_lr=.0002,
                                               bias_lr=.0003, output_scaling_lr=.0004)])
def test_optimizer_groups_apply_independent_learning_rates(overrides):
    model = convert_to_oi(make_model(), 1, [0.])
    groups = optimization_parameters(model, include_bias=True, **overrides)
    expected = dict(conductances=overrides.get("conductance_lr", .01),
                    significance=overrides.get("significance_lr", .0001),
                    bias=overrides.get("bias_lr", .0001),
                    output_scaling=overrides.get("output_scaling_lr", .0001))
    assert {g["name"]: g["lr"] for g in groups} == expected
    parameters = [p for g in groups for p in g["params"]]
    assert len({id(p) for p in parameters}) == len(parameters)
    # Equal gradients make Adam's first update approximately equal to each
    # group's LR, verifying that the returned groups actually drive updates.
    before = {id(p): p.detach().clone() for p in parameters}
    for p in parameters:
        p.grad = torch.ones_like(p)
    optimizer = torch.optim.Adam(groups)
    optimizer.step()
    for group in optimizer.param_groups:
        for p in group["params"]:
            torch.testing.assert_close(before[id(p)] - p, torch.full_like(p, group["lr"]),
                                       atol=2e-6, rtol=1e-3)


def test_optimizer_groups_omit_frozen_parameters():
    model = convert_to_oi(make_model(), 1, [0.])
    for tile in oi_tiles(model):
        tile.significance.requires_grad_(False)
        tile.out_scaling_alpha.requires_grad_(False)
    groups = optimization_parameters(model)
    assert [group["name"] for group in groups] == ["conductances"]


@pytest.mark.parametrize("rate", [-1., float("nan"), float("inf")])
def test_optimizer_groups_reject_invalid_learning_rates(rate):
    model = convert_to_oi(make_model(), 1, [0.])
    with pytest.raises(ValueError, match="learning rate"):
        optimization_parameters(model, significance_lr=rate)
