from copy import deepcopy

import pytest
import torch
from torch import nn

from aihwkit.inference.converter.conductance import (
    CustomPairConductanceConverter, NPairConductanceConverter,
)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.tiles.inference import InferenceTile

from experiments.common import (
    StockProgrammingEvaluator, evaluator_from_analog, evaluator_from_oi,
    extract_analog_programming, extract_oi_programming,
)
from ohm_improvement.conductance import convert_back_to_weights
from ohm_improvement.model import convert_to_oi
from ohm_improvement.tiles import OITile


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and (
        not torch.cuda.is_available() or not rpu_base.cuda.is_compiled()
    ):
        pytest.skip("CUDA-capable PyTorch and AIHWKIT required")
    return torch.device(request.param)


def make_model(device, digital_parameters=True):
    config = InferenceRPUConfig()
    config.forward.is_perfect = True
    config.mapping.weight_scaling_omega = 0.7 if digital_parameters else 0.
    config.mapping.learn_out_scaling = digital_parameters
    config.noise_model.g_converter = NPairConductanceConverter([1., 2.])
    model = nn.Sequential(
        nn.Sequential(nn.Linear(3, 4, bias=digital_parameters), nn.ReLU()),
        nn.Linear(4, 2, bias=digital_parameters),
    )
    return convert_to_analog(model, rpu_config=config).to(device).eval()


def tensors(scheme):
    mapping = scheme.conductances
    result = [*mapping.g_plus, *mapping.g_minus, mapping.significance, mapping.scale]
    return result + [t for t in (scheme.output_scaling, scheme.bias) if t is not None]


@pytest.mark.parametrize("digital_parameters", [False, True])
def test_oi_extraction_matches_analog_without_programming(device, digital_parameters):
    analog = make_model(device, digital_parameters)
    oi = convert_to_oi(analog, 1, [0.], noise_seeds=dict(train=11, validation=22, evaluation=33))
    stock_schemes = extract_analog_programming(analog)
    oi_schemes = extract_oi_programming(oi)
    assert stock_schemes.keys() == oi_schemes.keys()
    assert len(oi_schemes) == 2
    for name, scheme in oi_schemes.items():
        for actual, expected in zip(tensors(scheme), tensors(stock_schemes[name])):
            torch.testing.assert_close(actual, expected)
            assert not actual.requires_grad
            assert actual.grad_fn is None
            assert actual.device == device or (device.type == "cuda" and actual.device.type == "cuda")
        weights, _ = analog.get_submodule(name).get_weights(apply_weight_scaling=False)
        torch.testing.assert_close(convert_back_to_weights(scheme.conductances), weights.to(device))
        assert scheme.conductances.g_plus[0].shape == weights.shape
        assert (scheme.bias is not None) == digital_parameters
        assert (scheme.output_scaling is not None) == digital_parameters
    assert all(tile.active_weights is None for tile in oi.modules() if isinstance(tile, OITile))


def test_oi_extraction_preserves_optimized_pairs_and_owns_storage(device):
    oi = convert_to_oi(make_model(device), 1, [0.])
    tile = next(t for t in oi.modules() if isinstance(t, OITile))
    with torch.no_grad():
        tile.g_plus[0].add_(0.2)
        tile.g_minus[0].add_(0.2)
        tile.significance.mul_(1.1)
        tile.out_scaling_alpha.mul_(1.2)
        tile.bias.add_(0.1)
    schemes = extract_oi_programming(oi)
    name = next(name for name, t in oi.named_modules() if t is tile)
    scheme = schemes[name]
    expected = tile.g_plus[0].T if device.type == "cuda" else tile.g_plus[0]
    torch.testing.assert_close(scheme.conductances.g_plus[0], expected)
    saved = deepcopy(schemes)
    with torch.no_grad():
        for parameter in tile.parameters():
            parameter.zero_()
    for actual, expected in zip(tensors(scheme), tensors(saved[name])):
        torch.testing.assert_close(actual, expected)
    # Mutating the extracted snapshot also cannot change the model.
    for tensor in tensors(scheme):
        tensor.fill_(123.)
    assert all(torch.count_nonzero(p) == 0 for p in tile.parameters())


def test_analog_extraction_converter_override_preserves_source(device):
    model = make_model(device)
    original = deepcopy(model.state_dict())
    override = CustomPairConductanceConverter(
        f_lst=[2.], g_min=1., g_max=25.,
        g_lst=[[5., 10., 25.], [25., 10., 5.]],
    )
    schemes = extract_analog_programming(model, converter=override)
    for name, scheme in schemes.items():
        tile = model.get_submodule(name)
        assert isinstance(tile, InferenceTile)
        assert isinstance(tile.rpu_config.noise_model.g_converter, NPairConductanceConverter)
        weights, _ = tile.get_weights(apply_weight_scaling=False)
        expected, params = override.convert_to_conductances(weights)
        torch.testing.assert_close(scheme.conductances.g_plus[0], expected[0].to(device))
        torch.testing.assert_close(scheme.conductances.g_minus[0], expected[1].to(device))
        torch.testing.assert_close(scheme.conductances.scale, params["scale_ratio"].to(device))
        assert len(scheme.conductances.g_plus) == 1
        for tensor in tensors(scheme):
            assert not tensor.requires_grad
            tensor.zero_()
    # Re-extract to check independence from the modified snapshot.
    fresh = extract_analog_programming(model, converter=override)
    assert all(s.conductances.scale > 0 for s in fresh.values())
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, original[name])


def tile_weights(model):
    return {name: tile.get_weights(apply_weight_scaling=False)[0]
            for name, tile in model.named_modules() if isinstance(tile, InferenceTile)}


@pytest.mark.parametrize("compensation", [False, True])
def test_evaluator_noiseless_preserves_outputs_and_digital_parameters(device, compensation):
    analog = make_model(device)
    for tile in analog.analog_tiles():
        tile.rpu_config.noise_model.prog_noise_scale = 0.
        tile.rpu_config.noise_model.drift_scale = 0.
        tile.rpu_config.noise_model.read_noise_scale = 0.
        if not compensation:
            tile.rpu_config.drift_compensation = None
    oi = convert_to_oi(analog, 1, [0.])
    with torch.no_grad():
        for tile in oi.modules():
            if isinstance(tile, OITile):
                tile.bias.add_(0.1)
                tile.out_scaling_alpha.mul_(1.2)
    from ohm_improvement.model import program_weights
    program_weights(oi, 0, 0)
    x = torch.tensor([[0.2, -0.4, 0.8]], device=device)
    expected = oi(x).detach()
    mapping_scales = [tile.get_mapping_scales().clone() for tile in analog.analog_tiles()]
    evaluator = evaluator_from_oi(oi, analog, seed=123)
    assert evaluator.model is analog
    evaluator.program("test1")
    torch.testing.assert_close(analog(x), expected)
    evaluator.drift(86400.)
    torch.testing.assert_close(analog(x), expected)
    for tile, scales in zip(analog.analog_tiles(), mapping_scales):
        torch.testing.assert_close(tile.get_mapping_scales(), scales)


def test_evaluator_reuses_one_realization_and_is_reproducible(device, monkeypatch):
    model = make_model(device)
    evaluator = evaluator_from_analog(model, seed=123)
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    evaluator.program("test1")
    original_states = dict(evaluator._current_programmed)
    samples = {name: [g.clone() for g in state.g_plus + state.g_minus]
               for name, state in original_states.items()}
    # Drift must never call programming again.
    for tile in model.analog_tiles():
        def unexpected_programming(*args):
            raise AssertionError("Drift attempted to reprogram")
        monkeypatch.setattr(tile.rpu_config.noise_model,
                            "apply_programming_noise_to_conductance", unexpected_programming)
    evaluator.drift(86400.)
    expected = tile_weights(model)
    evaluator.drift(0.)
    evaluator.drift(365 * 86400.)
    evaluator.drift(86400.)
    for name, actual in tile_weights(model).items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
        assert evaluator._current_programmed[name] is original_states[name]
        for g, saved in zip(original_states[name].g_plus + original_states[name].g_minus, samples[name]):
            torch.testing.assert_close(g, saved, rtol=0, atol=0)
    monkeypatch.undo()
    evaluator.program("validation1")
    evaluator.drift(86400.)
    assert any(not torch.equal(w, expected[name]) for name, w in tile_weights(model).items())
    assert all(evaluator._current_programmed[name] is not state
               for name, state in original_states.items())
    evaluator.program("test1")
    evaluator.drift(86400.)
    for name, actual in tile_weights(model).items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))


def test_evaluator_matches_stock_conductance_methods_and_compensation(device):
    model = make_model(device)
    evaluator = evaluator_from_analog(model, seed=123)
    evaluator.program("test1")
    name, tile, scheme = evaluator._tiles[0]
    mapping = scheme.conductances
    noise = tile.rpu_config.noise_model
    state = evaluator._current_programmed[name]
    expected_drifted = []
    for side, targets, programmed, nus in (
        ("plus", mapping.g_plus, state.g_plus, state.drift_plus),
        ("minus", mapping.g_minus, state.g_minus, state.drift_minus),
    ):
        drifted = []
        for pair, target in enumerate(targets):
            with evaluator._rng(tile, evaluator._seed_for(name, "test1", "program", pair_index=pair, side=side)):
                expected = noise.apply_programming_noise_to_conductance(target)
            torch.testing.assert_close(programmed[pair], expected)
            with evaluator._rng(tile, evaluator._seed_for(name, "test1", "drift", pair_index=pair, side=side)):
                expected_nu = noise.generate_drift_coefficients(target)
            torch.testing.assert_close(nus[pair], expected_nu)
            with evaluator._rng(tile, evaluator._seed_for(name, "test1", "read", pair_index=pair, side=side, t_inference=86400.)):
                drifted.append(noise.apply_drift_noise_to_conductance(expected, expected_nu, 86400.))
        expected_drifted.append(drifted)
    from ohm_improvement.conductance import ConductanceMapping
    expected_weights = convert_back_to_weights(ConductanceMapping(
        *expected_drifted, mapping.significance, mapping.scale))
    evaluator.drift(86400.)
    torch.testing.assert_close(tile.get_weights(apply_weight_scaling=False)[0], expected_weights.cpu())
    compensation = tile.rpu_config.drift_compensation
    expected_alpha = compensation.apply(tile._forward_drift_readout_tensor(), state.compensation_baseline)
    torch.testing.assert_close(tile.alpha, expected_alpha)


def test_evaluator_validation_and_snapshot_ownership(device):
    model = make_model(device)
    schemes = extract_analog_programming(model)
    evaluator = StockProgrammingEvaluator(model, schemes, seed=123)
    for scheme in schemes.values():
        for t in tensors(scheme):
            t.zero_()
    evaluator.program("test1")
    assert all(torch.isfinite(w).all() and w.abs().sum() > 0
               for w in tile_weights(model).values())
    with pytest.raises(TypeError, match="string"):
        evaluator.program(1)
    for time in [-1., float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            evaluator.drift(time)
    with pytest.raises(ValueError, match="schemes"):
        StockProgrammingEvaluator(model, {}, seed=123)
    unprogrammed = evaluator_from_analog(model, seed=123)
    with pytest.raises(RuntimeError, match="program"):
        unprogrammed.drift(1.)



def test_evaluator_uses_custom_override_and_exact_reference(device):
    from aihwkit.inference.compensation.drift import GlobalDriftCompensationWithExactReference
    model = make_model(device, digital_parameters=False)
    converter = CustomPairConductanceConverter(
        f_lst=[2.], g_min=1., g_max=25.,
        g_lst=[[5., 10., 25.], [25., 10., 5.]],
    )
    expected = tile_weights(model)
    for tile in model.analog_tiles():
        tile.rpu_config.drift_compensation = GlobalDriftCompensationWithExactReference()
        tile.rpu_config.noise_model.prog_noise_scale = 0.
        tile.rpu_config.noise_model.drift_scale = 0.
        tile.rpu_config.noise_model.read_noise_scale = 0.
    evaluator = evaluator_from_analog(model, seed=123, converter=converter)
    evaluator.program("test1")
    evaluator.drift(86400.)
    for name, actual in tile_weights(model).items():
        torch.testing.assert_close(actual, expected[name])
        assert len(evaluator._current_programmed[name].g_plus) == 1



def test_scheme_detached_copy_handles_autograd_and_optional_parameters():
    from experiments.common import TileProgrammingScheme
    from ohm_improvement.conductance import ConductanceMapping
    source = torch.tensor([[2., 3.]], requires_grad=True)
    scheme = TileProgrammingScheme(
        ConductanceMapping([source * 2], [source * 0.5], source[0], source.sum()),
        output_scaling=source.sum().reshape(1), bias=source.sum().reshape(1),
    )
    copied = scheme.detached_copy(dtype=torch.float64)
    saved = [t.detach().double().clone() for t in tensors(scheme)]
    for actual, expected in zip(tensors(copied), saved):
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad and actual.grad_fn is None
    for tensor in tensors(scheme):
        with torch.no_grad():
            tensor.zero_()
    for actual, expected in zip(tensors(copied), saved):
        torch.testing.assert_close(actual, expected)
    minimal = TileProgrammingScheme(copied.conductances).detached_copy()
    assert minimal.bias is None and minimal.output_scaling is None
