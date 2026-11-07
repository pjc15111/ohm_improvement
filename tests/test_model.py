import pytest
import torch
from torch import nn

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import NPairConductanceConverter
from aihwkit.simulator.configs import InferenceRPUConfig

from ohm_improvement.model import OIModel, AnalogModel
from ohm_improvement.tiles import OITile


def prepare(model: OIModel, realization_id: str, time: float):
    model.program(realization_id)
    model.drift(time)


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
    converted = OIModel(original).eval()

    assert converted is not original
    assert len(oi_tiles(converted)) == 2
    assert not oi_tiles(original)
    assert all(tile.active_weights is None for tile in oi_tiles(converted))

    prepare(converted, "train1", 86400.0)
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
    model = OIModel(make_model())
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

    actual = model.optimization_parameters(**options)
    assert len(actual) == 1
    assert actual[0]["name"] == option.removeprefix("include_")
    assert [id(p) for p in actual[0]["params"]] == [id(p) for p in expected]


def test_absent_optional_parameters_are_not_returned():
    model = OIModel(make_model(bias=False, scaling=False))

    actual = model.optimization_parameters(include_conductances=False, include_output_scaling=True, include_bias=True, include_significance=False)
    assert actual == []


def test_model_clamping_reaches_every_tile():
    model = OIModel(make_model())
    with torch.no_grad():
        for tile in oi_tiles(model):
            for g in tile.g_plus:
                g.fill_(-2.0)
            for g in tile.g_minus:
                g.fill_(100.0)
            tile.significance.fill_(9.0)

    model.clamp()

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
    model = OIModel(make_model()).eval()
    prepare(model, "train1", 0.0)
    tiles = oi_tiles(model)

    # Export must work with live non-leaf tensors on the OITiles.
    assert all(tile.active_weights.grad_fn is not None for tile in tiles)
    active_ids = [id(tile.active_weights) for tile in tiles]

    x = torch.tensor([[0.2, -0.3, 0.8]])
    expected = model(x).detach().clone()
    exported = model.to_analog()

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


def test_standalone_analog_layer_is_supported():
    from aihwkit.nn import AnalogLinear
    analog = AnalogLinear(3, 2, rpu_config=InferenceRPUConfig()).eval()
    model = OIModel(analog)
    prepare(model, "test1", 86400.)
    assert isinstance(model.to_analog(), AnalogModel)


def test_model_seed_is_reproducible_and_distinct_per_tile():
    analog = make_model()
    rng = torch.random.get_rng_state().clone()
    first = OIModel(analog, seed=11)
    second = OIModel(analog, seed=11)
    assert torch.equal(rng, torch.random.get_rng_state())
    tiles = oi_tiles(first)
    assert not torch.equal(tiles[0].noise_seed, tiles[1].noise_seed)
    for a, b in zip(tiles, oi_tiles(second)):
        torch.testing.assert_close(a.noise_seed, b.noise_seed)
    with pytest.raises(TypeError, match="string"):
        first.program(1)


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        OIModel(make_model(), seed=seed)


def test_default_seed_follows_manual_seed():
    analog = make_model()
    torch.manual_seed(123)
    first = OIModel(analog)
    torch.manual_seed(123)
    second = OIModel(analog)
    for a, b in zip(oi_tiles(first), oi_tiles(second)):
        torch.testing.assert_close(a.noise_seed, b.noise_seed)


def test_digital_model_is_rejected():
    with pytest.raises(TypeError, match="AIHWKIT"):
        OIModel(nn.Linear(3, 2))


@pytest.mark.parametrize("overrides", [{}, dict(conductance_lr=.003, significance_lr=.0002,
                                               bias_lr=.0003, output_scaling_lr=.0004)])
def test_optimizer_groups_apply_independent_learning_rates(overrides):
    model = OIModel(make_model())
    groups = model.optimization_parameters(include_bias=True, **overrides)
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
    model = OIModel(make_model())
    for tile in oi_tiles(model):
        tile.significance.requires_grad_(False)
        tile.out_scaling_alpha.requires_grad_(False)
    groups = model.optimization_parameters()
    assert [group["name"] for group in groups] == ["conductances"]


@pytest.mark.parametrize("rate", [-1., float("nan"), float("inf")])
def test_optimizer_groups_reject_invalid_learning_rates(rate):
    model = OIModel(make_model())
    with pytest.raises(ValueError, match="learning rate"):
        model.optimization_parameters(significance_lr=rate)


def test_conversion_preserves_shared_analog_layers():
    from aihwkit.nn import AnalogLinear, AnalogSequential
    config = InferenceRPUConfig()
    config.forward.is_perfect = True
    layer = AnalogLinear(3, 3, rpu_config=config)
    model = OIModel(AnalogSequential(layer, layer))
    assert model._model[0] is model._model[1]
    assert len(list(model.oi_tiles())) == 1
    groups = model.optimization_parameters()
    params = [p for group in groups for p in group['params']]
    assert len(params) == len({id(p) for p in params})


def test_oi_checkpoint_restores_named_noise():
    analog = make_model()
    first = OIModel(analog, seed=11)
    first.program("test1")
    first.drift(86400.)
    restored = OIModel(analog, seed=99)
    restored.load_state_dict(first.state_dict())
    restored.program("test1")
    restored.drift(86400.)
    for a, b in zip(first.oi_tiles(), restored.oi_tiles()):
        torch.testing.assert_close(a.noise_seed, b.noise_seed)
        torch.testing.assert_close(a.active_weights, b.active_weights)


def test_tied_parameters_are_gathered_once():
    model = OIModel(make_model())
    first, second = list(model.oi_tiles())
    second.significance = first.significance
    groups = model.optimization_parameters()
    parameters = [parameter for group in groups for parameter in group['params']]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert sum(p is first.significance for p in parameters) == 1


def test_conflicting_parameter_groups_are_rejected():
    model = OIModel(make_model())
    tile = next(model.oi_tiles())
    tile.out_scaling_alpha = tile.significance
    with pytest.raises(ValueError, match="both.*significance.*output_scaling"):
        model.optimization_parameters()


def test_inplace_conversion_preflights_all_tiles():
    original = make_model()
    tiles = list(original.analog_tiles())
    weights = [tile.shared_weights for tile in tiles]
    compensations = [tile.rpu_config.drift_compensation for tile in tiles]
    tiles[-1].rpu_config.noise_model = None
    with pytest.raises(ValueError, match="noise model"):
        OIModel(original, seed=1, inplace=True)
    assert not any(isinstance(module, OITile) for module in original.modules())
    for tile, weight, compensation in zip(tiles, weights, compensations):
        assert tile.shared_weights is weight
        assert tile.rpu_config.drift_compensation is compensation


def test_model_dtype_movement_is_atomic():
    model = OIModel(make_model())
    model.program("test1")
    identities = [id(p) for p in model.parameters()]
    with pytest.raises(RuntimeError, match="before OI conversion"):
        model.double()
    assert identities == [id(p) for p in model.parameters()]
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(tile.get_analog_tile().shared_weights.dtype == torch.float32 for tile in model.oi_tiles())
    assert torch.isfinite(model(torch.ones(1, 3))).all()
