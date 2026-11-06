from copy import deepcopy

import pytest
import torch
from torch import nn

from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import NPairConductanceConverter
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.tiles.inference import InferenceTile

from ohm_improvement.tiles import OITile


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

    wrapped = OITile(deepcopy(original), 2, [0.0, 86400.0], None, noise_seeds=dict(train=11, validation=22, evaluation=33))
    wrapped.eval()
    wrapped.program_weights(0, 0)

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
    wrapped = OITile(make_tile(device), 1, [0.0], None, noise_seeds=dict(train=11, validation=22, evaluation=33))
    wrapped.eval()
    wrapped.program_weights(0, 0)

    x = torch.tensor([[0.2, 0.5, 0.9]], device=device, requires_grad=True)
    wrapped(x).sum().backward()

    for parameter in [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_clamping_preserves_parameters_and_enforces_bounds(device):
    wrapped = OITile(make_tile(device), 1, [0.0], None, noise_seeds=dict(train=11, validation=22, evaluation=33))
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
    wrapped = OITile(
        make_tile(device, compensation=compensation, noisy=True),
        2,
        [0.0, 86400.0],
        None,
        noise_seeds=dict(train=11, validation=22, evaluation=33),
    )
    wrapped.eval()
    with torch.no_grad():
        wrapped.significance.copy_(
            wrapped.significance.new_tensor([1.3, 2.7])
        )
        wrapped.bias.add_(0.1)
        wrapped.out_scaling_alpha.mul_(1.1)

    wrapped.program_weights(1, 1)
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


def test_accumulated_and_immediate_backward_agree(device):
    # Forward noise is disabled, but the fixed PCM samples remain active.
    wrapped = OITile(
        make_tile(device, compensation=True, noisy=True),
        2,
        [0.0, 86400.0],
        None,
        noise_seeds=dict(train=11, validation=22, evaluation=33),
    )
    wrapped.eval()
    parameters = [*wrapped.g_plus, *wrapped.g_minus, wrapped.significance]

    def gradients(accumulate):
        wrapped.zero_grad(set_to_none=True)
        x = torch.tensor(
            [[0.2, -0.4, 0.8]], device=device, requires_grad=True
        )
        losses = []
        for r, t in [(0, 0), (1, 1)]:
            wrapped.program_weights(r, t)
            # Reuse one programmed tile twice within a graph as well.
            loss = (
                wrapped(x).square().mean()
                + wrapped(0.7 * x).square().mean()
            ) / 2
            if accumulate:
                losses.append(loss)
            else:
                loss.backward()

        if accumulate:
            sum(losses).backward()

        return [p.grad.detach().clone() for p in parameters] + [
            x.grad.detach().clone()
        ]

    accumulated = gradients(True)
    immediate = gradients(False)

    for actual, expected in zip(accumulated, immediate):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)


def test_programming_validates_indices_and_requires_preparation():
    wrapped = OITile(make_tile(torch.device("cpu")), 2, [0.0], None, noise_seeds=dict(train=11, validation=22, evaluation=33))

    with pytest.raises(RuntimeError, match="program_weights"):
        wrapped(torch.ones(1, 3))
    with pytest.raises(RuntimeError, match="program_weights"):
        wrapped.to_analog()
    with pytest.raises(IndexError):
        wrapped.program_weights(2, 0)
    with pytest.raises(IndexError):
        wrapped.program_weights(0, 1)


def test_seeded_noise_streams_and_checkpoint(device):
    seeds = dict(train=11, validation=22, evaluation=33)
    wrapped = OITile(make_tile(device, noisy=True), 3, [0., 86400.], None,
                     noise_seeds=seeds)
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    sample = wrapped.noise_sample("program", 0, 0, 0)
    # Time and call order do not change programming/drift randomness.
    for kind in ("program", "drift"):
        torch.testing.assert_close(wrapped.noise_sample(kind, 0, 0, 0, 0),
                                   wrapped.noise_sample(kind, 0, 0, 0, 1), rtol=0, atol=0)
    others = [
        wrapped.noise_sample("program", 0, 0, 0, noise_split=split)
        for split in ("validation", "evaluation")
    ] + [wrapped.noise_sample("program", 1, 0, 0),
         wrapped.noise_sample("program", 0, 1, 0),
         wrapped.noise_sample("program", 0, 0, 1),
         wrapped.noise_sample("drift", 0, 0, 0)]
    assert all(not torch.equal(sample, other) for other in others)
    assert not torch.equal(wrapped.noise_sample("read", 0, 0, 0, 0),
                           wrapped.noise_sample("read", 0, 0, 0, 1))
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())

    wrapped.program_weights(1, 1, noise_split="evaluation")
    expected = wrapped.active_weights.detach().clone()
    restored = OITile(make_tile(device, noisy=True), 3, [0., 86400.], None, noise_seeds=dict(train=11, validation=22, evaluation=33))
    restored.load_state_dict(wrapped.state_dict())
    restored.program_weights(1, 1, noise_split="evaluation")
    torch.testing.assert_close(restored.active_weights, expected, rtol=0, atol=0)
    # Storage is constant in realization/time count, and samples stay out of checkpoints.
    assert set(wrapped._buffers) == {"scale", "noise_seeds"}
    wrapped.active_weights.square().sum().backward()
    for g in [*wrapped.g_plus, *wrapped.g_minus]:
        assert g.grad is not None and torch.isfinite(g.grad).all()
