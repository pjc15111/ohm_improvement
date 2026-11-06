"""Layer contracts needed by ResNet-32, a two-layer LSTM, and BERT/MNLI."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aihwkit.nn import (
    AnalogConv1d, AnalogConv2d, AnalogConv2dMapped, AnalogConv3d,
    AnalogConv1dMapped, AnalogConv3dMapped, AnalogLSTMCell, AnalogRNN,
)
from aihwkit.inference.converter.conductance import DualPairConductanceConverter
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.tiles.inference import InferenceTile

from ohm_improvement import OIModel


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and (not torch.cuda.is_available() or not rpu_base.cuda.is_compiled()):
        pytest.skip('CUDA-capable PyTorch and AIHWKIT required')
    return torch.device(request.param)


def config(split=(0, 0), noisy=False, scaling=False):
    cfg = InferenceRPUConfig()
    cfg.forward.is_perfect = True
    cfg.mapping.max_input_size, cfg.mapping.max_output_size = split
    cfg.mapping.weight_scaling_omega = .7 if scaling else 0.
    cfg.mapping.weight_scaling_columnwise = True
    cfg.mapping.learn_out_scaling = True
    cfg.mapping.out_scaling_columnwise = True
    if not noisy:
        cfg.noise_model.prog_noise_scale = 0.
        cfg.noise_model.drift_scale = 0.
        cfg.noise_model.read_noise_scale = 0.
    return cfg


def prepare(model, time=86400.):
    model.program('test1')
    model.drift(time)


def ordinary_convolutions(model):
    for layer in model.modules():
        if isinstance(layer, (AnalogConv2d, AnalogConv2dMapped)):
            layer.use_indexed = False


def compare_output_and_gradients(analog, inputs):
    """Compare against AIHWKIT, including native-layout conductance gradients."""
    ordinary_convolutions(analog)
    oi = OIModel(analog, seed=17).eval()
    prepare(oi)
    x = inputs.detach().clone().requires_grad_(inputs.is_floating_point())
    y = inputs.detach().clone().requires_grad_(inputs.is_floating_point())
    actual, expected = oi(x), analog(y)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    if x.requires_grad:
        torch.testing.assert_close(x.grad, y.grad, atol=5e-6, rtol=5e-4)
    assert_tile_gradients(oi, analog)
    torch.testing.assert_close(oi.to_analog()(inputs), actual.detach(), atol=3e-6, rtol=3e-5)
    return oi


def assert_tile_gradients(oi, analog):
    stock_tiles = {name: module for name, module in analog.named_modules() if isinstance(module, InferenceTile)}
    for name, tile in oi.named_oi_tiles():
        gradient = stock_tiles[name].shared_weights.grad
        assert gradient is not None
        for factor, gp, gm in zip(tile.significance, tile.g_plus, tile.g_minus):
            torch.testing.assert_close(gp.grad, gradient * factor / tile.scale, atol=5e-6, rtol=5e-4)
            torch.testing.assert_close(gm.grad, -gradient * factor / tile.scale, atol=5e-6, rtol=5e-4)


@pytest.mark.parametrize('split', [(0, 0), (5, 0), (0, 3), (5, 3)])
def test_mapped_linear_input_output_and_uneven_splits(device, split):
    analog = convert_to_analog(nn.Linear(7, 5), rpu_config=config(split)).to(device).eval()
    oi = compare_output_and_gradients(analog, torch.randn(2, 3, 7, device=device))
    assert len(list(oi.oi_tiles())) == (2 if split[0] else 1) * (2 if split[1] else 1)
    groups = oi.optimization_parameters(include_bias=True)
    biases = [p for group in groups if group['name'] == 'bias' for p in group['params']]
    assert len(biases) == 1
    assert biases[0].numel() == 5


@pytest.mark.parametrize('split', [(0, 0), (7, 3)])
@pytest.mark.parametrize('geometry', [dict(kernel_size=(2, 3), stride=(2, 1), padding=(1, 1)),
                                      dict(kernel_size=3, dilation=2, padding=2)])
def test_conv2d_unfold_geometry_and_mapped_arrays(device, split, geometry):
    digital = nn.Conv2d(2, 5, **geometry)
    analog = convert_to_analog(digital, rpu_config=config(split)).to(device).eval()
    oi = compare_output_and_gradients(analog, torch.randn(2, 2, 7, 9, device=device))
    assert oi._model.use_indexed is False
    for tile in oi.oi_tiles():
        assert tile.supports_indexed is False and not tile.is_indexed()
        assert not tile.get_analog_tile().get_analog_ctx().use_indexed
        assert tile.get_tensor_view(3) == (1, 1, -1)
    # Unfolding must adapt to new image shapes without a cached index map.
    x = torch.randn(1, 2, 8, 11, device=device)
    torch.testing.assert_close(oi(x), analog(x), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('split', [(0, 0), (9, 3)])
def test_legacy_mapped_conv2d(device, split):
    analog = AnalogConv2dMapped(3, 5, 3, padding=1, rpu_config=config(split)).to(device).eval()
    oi = compare_output_and_gradients(analog, torch.randn(2, 3, 5, 7, device=device))
    biases = [p for g in oi.optimization_parameters(include_bias=True) if g['name'] == 'bias' for p in g['params']]
    assert len(biases) == 1


@pytest.mark.parametrize('conv', [AnalogConv1d, AnalogConv3d, AnalogConv1dMapped, AnalogConv3dMapped])
def test_index_required_convolutions_are_rejected_before_mutation(conv):
    analog = conv(1, 2, 3, rpu_config=config())
    tiles = list(analog.analog_tiles())
    with pytest.raises(NotImplementedError, match='Conv2d'):
        OIModel(analog, inplace=True)
    assert list(analog.analog_tiles()) == tiles


def test_conversion_after_indexed_forward_clears_execution_mode(device):
    analog = AnalogConv2d(2, 3, 3, padding=1, use_indexed=True, rpu_config=config()).to(device).eval()
    x = torch.randn(2, 2, 5, 7, device=device)
    indexed = analog(x).detach()
    assert analog.analog_module.get_analog_ctx().use_indexed
    oi = OIModel(analog, seed=17).eval()
    prepare(oi)
    torch.testing.assert_close(oi(x), indexed, atol=3e-6, rtol=3e-5)
    assert not next(oi.oi_tiles()).get_analog_tile().get_analog_ctx().use_indexed
    assert analog.use_indexed is True  # Conversion must leave its source alone.
    exported = oi.to_analog()
    assert exported.use_indexed is False
    torch.testing.assert_close(exported(x), indexed, atol=3e-6, rtol=3e-5)


class ResidualBlock(nn.Module):
    """ResNet's stride-changing shortcut, batch norms, and in-place activations."""

    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(2, 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4),
        )
        self.shortcut = nn.Conv2d(2, 4, 1, stride=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch(x) + self.shortcut(x))


@pytest.mark.parametrize('split', [(0, 0), (9, 3)])
def test_resnet_residual_downsampling_and_batch_norm(device, split):
    analog = convert_to_analog(ResidualBlock(), rpu_config=config(split)).to(device).eval()
    oi = compare_output_and_gradients(analog, torch.randn(2, 2, 7, 9, device=device))
    assert isinstance(oi._model.branch[1], nn.BatchNorm2d)
    count = len(list(oi.oi_tiles()))
    assert count > 3 if split[0] else count == 3


@pytest.mark.parametrize('split', [(0, 0), (3, 5)])
def test_two_layer_lstm_gates_states_and_backpropagation_through_time(device, split):
    analog = AnalogRNN(
        AnalogLSTMCell, input_size=3, hidden_size=4, num_layers=2,
        rpu_config=config(split),
    ).to(device).eval()
    oi = OIModel(analog, seed=17).eval()
    prepare(oi)
    x = torch.randn(5, 2, 3, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, states = oi(x)
    expected, expected_states = analog(y)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert len(states) == 2
    for state, expected_state in zip(states, expected_states):
        for value, expected_value in zip(state, expected_state):
            torch.testing.assert_close(value, expected_value, atol=3e-6, rtol=3e-5)
    # Only the final output contributes: earlier inputs require recurrent gradients.
    actual[-1].square().sum().backward()
    expected[-1].square().sum().backward()
    assert x.grad[0].abs().sum() > 0
    torch.testing.assert_close(x.grad, y.grad, atol=5e-6, rtol=5e-4)
    assert_tile_gradients(oi, analog)
    snapshot_output, snapshot_states = oi.to_analog()(x.detach())
    torch.testing.assert_close(snapshot_output, actual.detach(), atol=3e-6, rtol=3e-5)
    # Carry both layers' hidden/cell states into a second sequence chunk.
    next_output, _ = oi(x.detach(), states)
    snapshot_next, _ = oi.to_analog()(x.detach(), snapshot_states)
    torch.testing.assert_close(next_output, snapshot_next, atol=3e-6, rtol=3e-5)


class TransformerBlock(nn.Module):
    """Small BERT-style encoder/classifier; attention stays digital."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.position = nn.Embedding(8, 8)
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)
        self.attention_output = nn.Linear(8, 8)
        self.norm1 = nn.LayerNorm(8)
        self.intermediate = nn.Linear(8, 12)
        self.output = nn.Linear(12, 8)
        self.norm2 = nn.LayerNorm(8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, ids):
        x = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        def heads(projection):
            return projection(x).view(*ids.shape, 2, 4).transpose(1, 2)
        scores = heads(self.query) @ heads(self.key).transpose(-1, -2) / math.sqrt(4)
        scores = scores.masked_fill((ids == 0)[:, None, None, :], float('-inf'))
        context = (scores.softmax(-1) @ heads(self.value)).transpose(1, 2).reshape_as(x)
        x = self.norm1(x + self.attention_output(context))
        x = self.norm2(x + self.output(F.gelu(self.intermediate(x))))
        return self.classifier(x[:, 0])


@pytest.mark.parametrize('split', [(0, 0), (5, 3)])
def test_bert_attention_sequence_projections_and_digital_layers(device, split):
    analog = convert_to_analog(TransformerBlock(), rpu_config=config(split)).to(device).eval()
    ids = torch.tensor([[1, 3, 4, 0], [1, 2, 5, 6]], device=device)
    oi = compare_output_and_gradients(analog, ids)
    # Digital operations must still propagate gradients through the analog layers.
    for name in ['embedding.weight', 'position.weight', 'norm1.weight', 'norm2.weight']:
        p, reference = dict(oi._model.named_parameters())[name], dict(analog.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0
        torch.testing.assert_close(p.grad, reference.grad, atol=5e-6, rtol=5e-4)
        assert all(p is not item for group in oi.optimization_parameters(include_bias=True) for item in group['params'])


def test_shared_attention_projection_keeps_alias_and_accumulates_gradients(device):
    analog = convert_to_analog(TransformerBlock(), rpu_config=config((5, 3))).to(device).eval()
    # OI must preserve aliases it receives, including reuse within one graph.
    analog.key = analog.query
    ids = torch.tensor([[1, 3, 4, 0], [1, 2, 5, 6]], device=device)
    oi = compare_output_and_gradients(analog, ids)
    assert oi._model.key is oi._model.query
    exported = oi.to_analog()
    assert exported.key is exported.query
    parameters = [p for group in oi.optimization_parameters(include_bias=True) for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})


@pytest.mark.parametrize('mapped', [False, True])
def test_conv_per_output_scaling_and_bias_match_stock(device, mapped):
    analog = AnalogConv2d(
        2, 5, 3, padding=1,
        rpu_config=config((7, 3) if mapped else (0, 0), scaling=True),
    ).to(device).eval()
    ordinary_convolutions(analog)
    with torch.no_grad():
        for tile in analog.analog_tiles():
            tile.out_scaling_alpha.copy_(torch.linspace(.6, 1.4, tile.out_size, device=device))
    oi = OIModel(analog, seed=17).eval()
    prepare(oi)
    x = torch.randn(2, 2, 5, 7, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = oi(x), analog(y)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=5e-6, rtol=5e-4)
    torch.testing.assert_close(oi.to_analog()(x.detach()), expected.detach(), atol=3e-6, rtol=3e-5)


def test_mapped_conv_scaling_bias_optimizer_checkpoint_and_noisy_export(device):
    cfg = config((7, 3), noisy=True, scaling=True)
    cfg.noise_model.g_converter = DualPairConductanceConverter(f_lst=[1., 2.])
    analog = AnalogConv2d(2, 5, 3, padding=1, rpu_config=cfg).to(device).eval()
    oi = OIModel(analog, seed=17).eval()
    x = torch.randn(2, 2, 5, 7, device=device)
    optimizer = torch.optim.Adam(oi.optimization_parameters(include_bias=True))
    prepare(oi)
    losses = []
    for time in [86400., 31536000.]:
        oi.drift(time)
        losses.append(oi(x).square().mean())
    sum(losses).backward()
    for group in optimizer.param_groups:
        assert all(p.grad is not None and p.grad.isfinite().all() for p in group['params'])
    optimizer.step()
    oi.clamp()
    prepare(oi)
    expected = oi(x).detach()
    torch.testing.assert_close(oi.to_analog()(x), expected, atol=3e-6, rtol=3e-5)
    oi.drift(1.)
    oi.drift(86400.)
    torch.testing.assert_close(oi(x), expected, atol=3e-6, rtol=3e-5)
    restored = OIModel(analog, seed=99).eval()
    restored.load_state_dict(oi.state_dict())
    prepare(restored)
    torch.testing.assert_close(restored(x), expected, atol=3e-6, rtol=3e-5)
    for tile in restored.oi_tiles():
        with pytest.raises(NotImplementedError, match='indexing'):
            tile.set_indexed(torch.tensor([0]), (5, 7))
