from copy import deepcopy

import pytest
import torch
from torch import nn

from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import (
    NPairConductanceConverter, CustomPairConductanceConverter,
)
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.tiles.inference import InferenceTile

from ohm_improvement.tiles import OITile


def prepare(model: OITile, realization_id: str, time: float):
    model.program(realization_id)
    model.drift(time)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available() or not rpu_base.cuda.is_compiled():
            pytest.skip("CUDA-capable PyTorch and AIHWKIT required")
    return torch.device(request.param)


def make_tile(device, compensation=False, noisy=False):
    config = InferenceRPUConfig()
    config.forward.is_perfect = True
    config.drift_compensation = (
        GlobalDriftCompensation() if compensation else None
    )
    config.mapping.weight_scaling_omega = 0.7
    config.mapping.learn_out_scaling = True
    config.noise_model.g_converter = NPairConductanceConverter([1.0, 2.0])

    if not noisy:
        config.noise_model.prog_noise_scale = 0.0
        config.noise_model.drift_scale = 0.0
        config.noise_model.read_noise_scale = 0.0

    tile = InferenceTile(2, 3, config, bias=True).to(device)
    tile.set_weights(
        torch.tensor([[0.3, -0.2, 0.1], [-0.4, 0.15, 0.25]]),
        torch.tensor([0.07, -0.09]),
    )
    tile.eval()
    return tile


@pytest.mark.parametrize("compensation", [False, True])
def test_initialization_preserves_backend_weights(device, compensation):
    original = make_tile(device, compensation=compensation)
    raw, _ = original.get_weights(apply_weight_scaling=False)
    assert raw.abs().max() > 0

    wrapped = OITile(deepcopy(original), None, seed=11)
    wrapped.eval()
    prepare(wrapped, "train1", 0.)

    expected_weights = raw.to(device)
    if device.type == "cuda":
        expected_weights = expected_weights.T
    torch.testing.assert_close(
        wrapped.active_weights, expected_weights, atol=1e-6, rtol=1e-5
    )

    # Also exercise arbitrary leading dimensions, scales, and digital bias.
    x = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]], device=device
    )
    torch.testing.assert_close(
        wrapped(x), original(x), atol=2e-6, rtol=2e-5
    )


def test_gradients_reach_every_pair_and_significance(device):
    wrapped = OITile(make_tile(device), None, seed=11)
    wrapped.eval()
    prepare(wrapped, "train1", 0.)

    x = torch.tensor([[0.2, 0.5, 0.9]], device=device, requires_grad=True)
    wrapped(x).sum().backward()

    for parameter in [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_clamping_preserves_parameters_and_enforces_bounds(device):
    wrapped = OITile(make_tile(device), None, seed=11)
    parameters = [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]
    identities = [id(p) for p in parameters]

    with torch.no_grad():
        for g in [*wrapped.g_plus, *wrapped.g_minus]:
            g.flatten()[0] = -10.0
            g.flatten()[1] = 100.0
        wrapped.significance.copy_(wrapped.significance.new_tensor([-1.0, 9.0]))

    wrapped.clamp_conductances()
    wrapped.clamp_significance()

    assert identities == [
        id(p) for p in [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]
    ]
    for g in [*wrapped.g_plus, *wrapped.g_minus]:
        assert (g >= wrapped.g_converter.g_min).all()
        assert (g <= wrapped.g_converter.g_max).all()
    torch.testing.assert_close(
        wrapped.significance,
        wrapped.significance.new_tensor([0.1, 5.0]),
    )


@pytest.mark.parametrize("compensation", [False, True])
def test_export_matches_snapshot_and_is_independent(device, compensation):
    wrapped = OITile(make_tile(device, compensation=compensation, noisy=True), None, seed=11)
    wrapped.eval()
    with torch.no_grad():
        wrapped.significance.copy_(
            wrapped.significance.new_tensor([1.3, 2.7])
        )
        wrapped.bias.add_(0.1)
        wrapped.out_scaling_alpha.mul_(1.1)

    prepare(wrapped, "train2", 86400.)
    x = torch.tensor([[0.2, -0.4, 0.8]], device=device)
    expected = wrapped(x).detach().clone()
    active = wrapped.active_weights

    exported = wrapped.to_analog()

    assert isinstance(exported.shared_weights, nn.Parameter)
    assert wrapped.active_weights is active
    torch.testing.assert_close(exported(x), expected, atol=2e-6, rtol=2e-5)
    assert exported.rpu_config.noise_model.g_converter.f_lst == pytest.approx(
        [1.3, 2.7]
    )

    with torch.no_grad():
        exported.bias.add_(1.0)
    torch.testing.assert_close(wrapped(x), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("compensation", [False, True])
def test_accumulated_and_immediate_backward_agree(device, compensation):
    # Forward noise is disabled, but the fixed PCM samples remain active.
    wrapped = OITile(make_tile(device, compensation=compensation, noisy=True), None, seed=11)
    wrapped.eval()
    parameters = [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]

    def gradients(accumulate):
        wrapped.zero_grad(set_to_none=True)
        x = torch.tensor(
            [[0.2, -0.4, 0.8]], device=device, requires_grad=True
        )
        losses = []
        snapshots = []
        for r, t in [(0, 0), (1, 1)]:
            prepare(wrapped, f"train{r}", [0., 86400.][t])
            snapshots.append(wrapped.active_weights.detach().clone())
            # Reuse one programmed tile twice within a graph as well.
            loss = (
                wrapped(x).square().mean()
                + wrapped(0.7 * x).square().mean()
            ) / 2
            snapshots.append(loss.detach().clone())
            if accumulate:
                losses.append(loss)
            else:
                loss.backward()

        if accumulate:
            sum(losses).backward()

        return snapshots, [p.grad.detach().clone() for p in parameters] + [
            x.grad.detach().clone()
        ]

    accumulated_forward, accumulated = gradients(True)
    immediate_forward, immediate = gradients(False)

    for index, (actual, expected) in enumerate(zip(accumulated_forward, immediate_forward)):
        torch.testing.assert_close(
            actual, expected, atol=2e-6, rtol=2e-4,
            msg=lambda message: f"Forward snapshot {index}: {message}",
        )
    names = [f"g_plus[{i}]" for i in range(len(wrapped.g_plus))]
    names += [f"g_minus[{i}]" for i in range(len(wrapped.g_minus))]
    names += ["significance", "input"]
    for name, actual, expected in zip(names, accumulated, immediate):
        torch.testing.assert_close(
            actual, expected, atol=2e-6, rtol=2e-4,
            msg=lambda message, name=name: (
                f"{name} gradient (compensation={compensation}): {message}\n"
                f"Accumulated: {actual.detach().cpu().tolist()}\n"
                f"Immediate: {expected.detach().cpu().tolist()}"
            ),
        )


def test_programming_requires_named_realization_and_preparation():
    wrapped = OITile(make_tile(torch.device("cpu")), seed=11)
    for operation in (lambda: wrapped(torch.ones(1, 3)), wrapped.to_analog,
                      lambda: wrapped.drift(0.)):
        with pytest.raises(RuntimeError, match="program"):
            operation()
    with pytest.raises(TypeError, match="string"):
        wrapped.program(1)
    wrapped.program("validation1")
    for time in (-1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            wrapped.drift(time)


def test_named_noise_and_checkpoint(device):
    wrapped = OITile(make_tile(device, noisy=True), seed=11)
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    sample = wrapped.noise_sample("program", 0, "plus", "train1")
    for kind in ("program", "drift"):
        torch.testing.assert_close(wrapped.noise_sample(kind, 0, "plus", "train1", 0.),
                                   wrapped.noise_sample(kind, 0, "plus", "train1", 86400.), rtol=0, atol=0)
    others = [wrapped.noise_sample("program", 0, "plus", name)
              for name in ("validation1", "test1", "train2")]
    others += [wrapped.noise_sample("program", 1, "plus", "train1"),
               wrapped.noise_sample("program", 0, "minus", "train1"),
               wrapped.noise_sample("drift", 0, "plus", "train1")]
    assert all(not torch.equal(sample, other) for other in others)
    assert not torch.equal(wrapped.noise_sample("read", 0, "plus", "train1", 0.),
                           wrapped.noise_sample("read", 0, "plus", "train1", 86400.))
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    with pytest.raises(ValueError, match="side"):
        wrapped.noise_sample("read", 0, 0, "train1")
    prepare(wrapped, "test1", 86400.)
    expected = wrapped.active_weights.detach().clone()
    restored = OITile(make_tile(device, noisy=True), seed=99)
    restored.load_state_dict(wrapped.state_dict())
    prepare(restored, "test1", 86400.)
    torch.testing.assert_close(restored.active_weights, expected, rtol=0, atol=0)
    assert set(wrapped._buffers) == {"scale", "noise_seed"}
    wrapped.active_weights.square().sum().backward()
    for g in [*wrapped.g_plus, *wrapped.g_minus]:
        assert g.grad is not None and torch.isfinite(g.grad).all()


def test_custom_converter_initialization_and_gradients(device):
    original = make_tile(device)
    converter = CustomPairConductanceConverter(
        f_lst=[2.0], g_min=1., g_max=25.,
        g_lst=[[5., 10., 25.], [25., 10., 5.]],
    )
    raw, _ = original.get_weights(apply_weight_scaling=False)
    native_weights = raw.to(device)
    if device.type == "cuda":
        native_weights = native_weights.T
    conductances, _ = converter.convert_to_conductances(native_weights.cpu())
    wrapped = OITile(original, converter, seed=11)
    torch.testing.assert_close(wrapped.g_plus[0], conductances[0].to(device))
    torch.testing.assert_close(wrapped.g_minus[0], conductances[1].to(device))
    prepare(wrapped, "train1", 0.)
    torch.testing.assert_close(wrapped.active_weights, native_weights)
    wrapped(torch.tensor([[0.2, -0.4, 0.8]], device=device)).square().sum().backward()
    for parameter in [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("compensation", [False, True])
def test_first_backend_binding_does_not_overwrite_programmed_weights(monkeypatch, compensation):
    tile = make_tile(torch.device("cpu"), compensation=compensation, noisy=True)
    backend_weights = tile.get_weights(apply_weight_scaling=False)[0].clone()
    original_ensure = tile.ensure_shared_weights
    first_binding = True

    def lazy_cuda_binding(shared_weights=None):
        # Reproduce CUDA's destructive first binding on CPU so this regression
        # can also be exercised without a GPU.
        nonlocal first_binding
        if first_binding:
            destination = tile.shared_weights if shared_weights is None else shared_weights
            with torch.no_grad():
                destination.copy_(backend_weights)
            first_binding = False
        original_ensure(shared_weights)

    monkeypatch.setattr(tile, "ensure_shared_weights", lazy_cuda_binding)
    wrapped = OITile(tile, None, seed=11)
    assert not first_binding
    prepare(wrapped, "train1", 0.)
    expected = wrapped.active_weights.detach().clone()
    x = torch.tensor([[0.2, -0.4, 0.8]])
    first = wrapped(x)
    torch.testing.assert_close(wrapped.active_weights, expected, rtol=0, atol=0)
    prepare(wrapped, "train1", 0.)
    torch.testing.assert_close(wrapped(x), first, rtol=0, atol=0)


@pytest.mark.parametrize("compensation", [False, True])
def test_reused_programming_graph_matches_reprogramming(device, compensation):
    wrapped = OITile(make_tile(device, compensation=compensation, noisy=True), seed=17)
    times = [0., 86400., 365 * 86400.]

    def gradients(reprogram):
        wrapped.zero_grad(set_to_none=True)
        x = torch.tensor([[.2, -.4, .8]], device=device, requires_grad=True)
        wrapped.program("train1")
        state = wrapped._programmed
        outputs = []
        for index, time in enumerate(times):
            if reprogram:
                wrapped.program("train1")
            wrapped.drift(time)
            output = wrapped(x)
            outputs.append(output.detach().clone())
            output.square().sum().backward(retain_graph=not reprogram and index < len(times) - 1)
            if not reprogram:
                assert wrapped._programmed is state
        return outputs, [p.grad.detach().clone() for p in
                         [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]] + [x.grad.clone()]

    outputs, reused = gradients(False)
    expected_outputs, fresh = gradients(True)
    for actual, expected in zip(outputs + reused, expected_outputs + fresh):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)


def test_drift_reuses_state_and_rejects_parameter_updates(device):
    wrapped = OITile(make_tile(device, noisy=True), seed=17)
    wrapped.program("test1")
    state = wrapped._programmed
    wrapped.drift(86400.)
    expected = wrapped.active_weights.detach().clone()
    wrapped.drift(0.)
    wrapped.drift(365 * 86400.)
    wrapped.drift(86400.)
    assert wrapped._programmed is state
    torch.testing.assert_close(wrapped.active_weights, expected, rtol=0, atol=0)
    with torch.no_grad():
        wrapped.g_plus[0].add_(.1)
    with pytest.raises(RuntimeError, match="Parameters changed"):
        wrapped.drift(0.)
    with pytest.raises(RuntimeError, match="Parameters changed"):
        wrapped(torch.ones(1, 3, device=device))
    wrapped.program("train2")
    assert wrapped._programmed is not state
    wrapped.drift(0.)


@pytest.mark.parametrize("compensation", [False, True])
@pytest.mark.parametrize("drift", [False, True])
def test_export_before_first_forward(device, compensation, drift):
    wrapped = OITile(make_tile(device, compensation=compensation, noisy=True), seed=17)
    wrapped.program("test1")
    if drift:
        wrapped.drift(86400.)
    x = torch.tensor([[.2, -.4, .8]], device=device)
    torch.testing.assert_close(wrapped.to_analog()(x), wrapped(x))


@pytest.mark.parametrize("mapping", ["none", "tile", "column"])
@pytest.mark.parametrize("lr_compensation", [False, True])
@pytest.mark.parametrize("compensation", [False, True])
def test_gradients_match_torch_reference(device, mapping, lr_compensation, compensation):
    """Compare mathematical derivatives, independently of AIHWKIT update gradients."""
    config = InferenceRPUConfig()
    config.forward.is_perfect = True
    config.mapping.weight_scaling_omega = 0. if mapping == "none" else .7
    config.mapping.weight_scaling_columnwise = mapping == "column"
    config.mapping.weight_scaling_lr_compensation = lr_compensation
    config.mapping.learn_out_scaling = True
    config.mapping.out_scaling_columnwise = True
    config.noise_model.g_converter = NPairConductanceConverter([1., 2.])
    config.drift_compensation = GlobalDriftCompensation() if compensation else None
    original = InferenceTile(2, 3, config, bias=True).to(device)
    original.set_weights(torch.tensor([[.3, -.2, .1], [-.7, .15, .25]]), torch.tensor([.07, -.09]))
    wrapped = OITile(original, seed=11).eval()
    wrapped.program("reference")
    programmed = wrapped.active_weights
    wrapped.drift(86400.)
    x = torch.tensor([[[.2, -.4, .8], [.7, .1, -.3]]], device=device, requires_grad=True)
    tile = wrapped.get_analog_tile()
    scales = tile.get_mapping_scales()
    if scales is None:
        scales = x.new_ones(1)

    def matmul(inputs, weights):
        return inputs @ (weights if tile.is_cuda else weights.T) * scales

    expected = matmul(x, wrapped.active_weights)
    if compensation:
        probe = torch.eye(3, device=device)
        reference = matmul(probe, programmed).abs().mean().clamp(min=1e-4)
        current = matmul(probe, wrapped.active_weights).abs().mean().clamp(min=1e-4)
        expected = expected * (reference / current)
    expected = expected * wrapped.out_scaling_alpha + wrapped.bias
    actual = wrapped(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    parameters = [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance,
                  wrapped.bias, wrapped.out_scaling_alpha, x]
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("assign", [False, True])
def test_checkpoint_loading_invalidates_programming(device, assign):
    wrapped = OITile(make_tile(device, compensation=True), seed=17)
    prepare(wrapped, "test1", 0.)
    state = {name: value.detach().clone() for name, value in wrapped.state_dict().items()}
    state["significance"] *= 2
    wrapped.load_state_dict(state, assign=assign)
    with pytest.raises(RuntimeError, match="Call program"):
        wrapped(torch.ones(1, 3, device=device))
    with pytest.raises(RuntimeError, match="Call program"):
        wrapped.drift(0.)
    prepare(wrapped, "test1", 0.)
    assert torch.isfinite(wrapped(torch.ones(1, 3, device=device))).all()


def test_replacing_parameters_or_buffers_invalidates_programming():
    wrapped = OITile(make_tile(torch.device("cpu")), seed=17)
    for name in ["significance", "scale", "noise_seed"]:
        wrapped.program("test1")
        value = getattr(wrapped, name).detach().clone()
        setattr(wrapped, name, nn.Parameter(value) if name == "significance" else value)
        with pytest.raises(RuntimeError, match="Parameters changed"):
            wrapped.drift(0.)


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_invalid_seed_does_not_mutate_tile(seed):
    tile = make_tile(torch.device("cpu"), compensation=True)
    compensation = tile.rpu_config.drift_compensation
    weights = tile.shared_weights
    tile.get_analog_ctx().set_indexed(True)
    with pytest.raises(ValueError, match="seed"):
        OITile(tile, seed=seed)
    assert tile.shared_weights is weights
    assert tile.rpu_config.drift_compensation is compensation
    assert tile.get_analog_ctx().use_indexed


@pytest.mark.parametrize("unsupported", ["bias", "noise", "missing_noise", "compensation"])
def test_unsupported_configuration_does_not_mutate_tile(unsupported):
    config = InferenceRPUConfig()
    config.mapping.digital_bias = unsupported != "bias"
    tile = InferenceTile(2, 3, config, bias=True)
    if unsupported == "noise":
        tile.rpu_config.noise_model = object()
    elif unsupported == "missing_noise":
        tile.rpu_config.noise_model = None
    elif unsupported == "compensation":
        tile.rpu_config.drift_compensation = object()
    compensation = tile.rpu_config.drift_compensation
    weights = tile.shared_weights
    tile.get_analog_ctx().set_indexed(True)
    with pytest.raises((ValueError, TypeError), match="bias|noise|compensation"):
        OITile(tile, seed=1)
    assert tile.shared_weights is weights
    assert tile.rpu_config.drift_compensation is compensation
    assert tile.get_analog_ctx().use_indexed


def test_compensation_calibrates_once_per_program_and_drift(monkeypatch, device):
    wrapped = OITile(make_tile(device, compensation=True, noisy=True), seed=1)
    calls = []
    forward = wrapped.analog_tile_holder.forward

    def count_forward(*args, **kwargs):
        calls.append(args[0].shape)
        return forward(*args, **kwargs)

    monkeypatch.setattr(wrapped.analog_tile_holder, "forward", count_forward)
    wrapped.program("test1")
    assert len(calls) == 1
    probe = wrapped._probe
    wrapped.drift(86400.)
    assert len(calls) == 2
    factor = wrapped._compensation_factor
    x = torch.ones(1, 3, device=device)
    first = wrapped(x)
    second = wrapped(x)
    assert len(calls) == 4
    assert wrapped._compensation_factor is factor
    exported = wrapped.to_analog()
    assert len(calls) == 4  # Export preserves calibration rather than resampling it.
    torch.testing.assert_close(exported.alpha, factor)
    torch.testing.assert_close(first, second)
    (first.sum() + second.sum()).backward()
    wrapped.program("test2")
    wrapped.drift(86400.)
    assert wrapped._probe is probe
    assert wrapped._compensation_factor is not factor


def test_dtype_movement_rejected_before_mutation():
    wrapped = OITile(make_tile(torch.device("cpu")), seed=17)
    wrapped.program("test1")
    identities = [id(p) for p in wrapped.parameters()]
    with pytest.raises(RuntimeError, match="before OI conversion"):
        wrapped.double()
    assert identities == [id(p) for p in wrapped.parameters()]
    assert all(p.dtype == torch.float32 for p in wrapped.parameters())
    wrapped.to("cpu")
    assert torch.isfinite(wrapped(torch.ones(1, 3))).all()


@pytest.mark.parametrize("compensation", [False, True])
def test_conductance_and_significance_gradients_match_finite_differences(device, compensation):
    wrapped = OITile(make_tile(device, compensation=compensation), seed=19).eval()
    # Stay away from conductance clamps for a two-sided derivative.
    with torch.no_grad():
        for parameter in [*wrapped.g_plus, *wrapped.g_minus]:
            parameter.add_(2.)
    x = torch.tensor([[.2, -.4, .8]], device=device)

    def objective():
        prepare(wrapped, "finite-difference", 86400.)
        return wrapped(x).square().sum()

    parameters = [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]
    gradients = torch.autograd.grad(objective(), parameters)
    for parameter, gradient in zip(parameters, gradients):
        epsilon = .01 if parameter.ndim == 2 else .001
        with torch.no_grad():
            original = parameter.flatten()[0].item()
            parameter.flatten()[0] = original + epsilon
            plus = objective().item()
            parameter.flatten()[0] = original - epsilon
            minus = objective().item()
            parameter.flatten()[0] = original
        expected = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close(gradient.flatten()[0], gradient.new_tensor(expected),
                                   atol=2e-5, rtol=5e-3)


def test_failed_converter_does_not_mutate_source_tile():
    tile = make_tile(torch.device("cpu"), compensation=True)
    weights = tile.shared_weights
    compensation = tile.rpu_config.drift_compensation

    class BrokenConverter:
        def convert_to_conductances(self, weights):
            return [torch.ones(1, 2), torch.ones(2, 1)], {"scale_ratio": torch.tensor(1.)}

    with pytest.raises(ValueError, match="same shape"):
        OITile(tile, BrokenConverter(), seed=1)
    assert tile.shared_weights is weights
    assert tile.rpu_config.drift_compensation is compensation
