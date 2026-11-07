"""MNIST HAT, stock WPO, progress reporting, and final evaluation."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import random
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator import rpu_base
from aihwkit.simulator.parameters.enums import WeightClipType, WeightModifierType

from experiments.mnist import common, hat, wpo


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and (not torch.cuda.is_available() or not rpu_base.cuda.is_compiled()):
        pytest.skip('CUDA-capable PyTorch and AIHWKIT required')
    return torch.device(request.param)


def test_hat_config_and_cli_defaults():
    args = wpo.parse_args([])
    assert args.hat_epochs == 0
    assert args.checkpoint == 'results/mnist/digital.pt'
    args = wpo.parse_args(['--dry-run', '--hat-epochs', '3'])
    assert args.hat_epochs == args.evaluation_realizations == len(args.times) == 1
    assert args.wpo_maxiter == 1 and not args.wpo_polish and args.workers == 1
    cfg = hat.make_hat_config([1., 2.], weight_noise=.06, clip_sigma=3.)
    assert cfg.modifier.type == WeightModifierType.ADD_NORMAL
    assert cfg.modifier.std_dev == .06 and not cfg.modifier.enable_during_test
    assert cfg.clip.type == WeightClipType.LAYER_GAUSSIAN and cfg.clip.sigma == 3.
    assert cfg.noise_model.g_converter.f_lst == [1., 2.]
    assert cfg.mapping == common.make_rpu_config([1., 2.]).mapping


@pytest.mark.parametrize('options', [
    ['--hat-epochs', '-1'], ['--batch-size', '0'], ['--hat-lr', 'nan'],
    ['--hat-weight-noise', '-1'], ['--times', 'inf'], ['--noise-seed', '-1'],
    ['--workers', '0'], ['--wpo-bins', '1'], ['--wpo-input-size', '0'],
    ['--evaluation-interval', '-1'], ['--wpo-loss-margin', '1'],
])
def test_invalid_preparation_options(options):
    with pytest.raises(SystemExit):
        wpo.parse_args(options)


@pytest.mark.parametrize('epochs', [0, 2])
def test_hat_updates_copy_records_steps_and_returns_clean_cpu_model(device, epochs):
    cfg = common.make_rpu_config([1.])
    cfg.forward.is_perfect = True
    analog = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(dim=-1)), cfg).eval()
    before = [tuple(value.clone() for value in tile.get_weights()) for tile in analog.analog_tiles()]
    args = wpo.parse_args(['--hat-epochs', str(epochs), '--hat-weight-noise', '0',
                          '--hat-clip-sigma', '100'])
    loader = DataLoader(TensorDataset(
        torch.tensor([[.2, -.4, .8], [.1, .3, -.2], [.4, .2, .1]]),
        torch.tensor([0, 1, 0]),
    ), batch_size=2)
    trained, history = hat.train_hardware_aware(args, analog, loader, loader, device)
    assert trained is not analog and not trained.training
    for tile, initial in zip(analog.analog_tiles(), before):
        for actual, expected in zip(tile.get_weights(), initial):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert tile.rpu_config.modifier.type == WeightModifierType.NONE
    for tile in trained.analog_tiles():
        assert not tile.is_cuda
        assert tile.rpu_config.modifier.type == WeightModifierType.NONE
        assert tile.rpu_config.clip.type == WeightClipType.NONE
    if epochs == 0:
        assert history == []
        for tile, initial in zip(trained.analog_tiles(), before):
            for actual, expected in zip(tile.get_weights(), initial):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        assert any(not torch.equal(tile.get_weights()[0], initial[0])
                   for tile, initial in zip(trained.analog_tiles(), before))
        records = [batch for epoch in history for batch in epoch['training']['batches']]
        assert [r.step for r in records] == [1, 2, 3, 4]
        assert [r.epoch for r in records] == [.5, 1., 1.5, 2.]
        assert [r.seconds for r in records] == sorted(r.seconds for r in records)
        for epoch in history:
            assert epoch['phase'] == 'hat'
            assert epoch['training']['count'] == epoch['evaluation']['count'] == 3
            assert abs(sum(r.metrics['loss'] * r.count for r in epoch['training']['batches'])
                       / 3 - epoch['training']['loss']) < 1e-6


@pytest.mark.parametrize('weight_noise,clip_sigma', [(.2, 100.), (0., .1)])
def test_hat_training_noise_and_optimizer_clipping_are_active(device, weight_noise, clip_sigma):
    cfg = hat.make_hat_config([1.], weight_noise=weight_noise, clip_sigma=clip_sigma)
    cfg.forward.is_perfect = True
    digital = nn.Sequential(nn.Linear(3, 2, bias=False), nn.LogSoftmax(-1))
    with torch.no_grad():
        digital[0].weight.copy_(torch.tensor([[-.5, .1, .8], [.9, -.4, .3]]))
    model = convert_to_analog(digital, cfg).to(device)
    tile = next(model.analog_tiles())
    before = tile.get_weights()[0].clone()
    # Freeze learning so any permanent change comes from clipping alone.
    optimizer = AnalogSGD(model.parameters(), lr=0.)
    optimizer.regroup_param_groups(model)
    x = torch.tensor([[.2, -.4, .8], [.1, .3, -.2]], device=device)
    loader = DataLoader(TensorDataset(x, torch.tensor([0, 1], device=device)), batch_size=2)
    hat.train_hat_epoch(model, loader, device, optimizer, log_interval=10)
    reference = tile.get_weights()[0].to(device)
    with torch.no_grad():
        noisy = model.train()(x)
        clean = model.eval()(x)
    torch.testing.assert_close(clean, F.log_softmax(F.linear(x, reference), -1), atol=3e-6, rtol=3e-5)
    if weight_noise:
        torch.testing.assert_close(reference.cpu(), before, atol=0, rtol=0)
        assert not torch.allclose(noisy, clean)
    else:
        assert reference.abs().max() < before.abs().max()
        torch.testing.assert_close(noisy, clean, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('hat_epochs', [0, 1])
def test_full_command_runs_stock_wpo_and_saves_results(monkeypatch, tmp_path, hat_epochs):
    def fake_mnist(root, train, download, transform):
        return TensorDataset(torch.zeros(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr(common.datasets, 'MNIST', fake_mnist)
    checkpoint, results = tmp_path / 'baseline.pt', tmp_path / 'history.json'
    weights = common.Net().state_dict()
    torch.save(weights, checkpoint)
    history = wpo.main([
        '--checkpoint', str(checkpoint), '--results-file', str(results),
        '--no-accel', '--dry-run', '--batch-size', '2', '--test-batch-size', '2',
        '--hat-epochs', str(hat_epochs),
    ])
    assert history.method == 'wpo' and history.config['stage'] == 'complete'
    assert history.digital_baseline['count'] == 2
    assert len([r for r in history.optimization_history if isinstance(r, dict)]) == hat_epochs
    records = [r for r in history.optimization_history if isinstance(r, wpo.HistoryRecord)]
    assert any('objective' in r.metrics for r in records)
    assert any(r.realization_id == 'validation1' for r in records)
    assert records[-1].metrics['evaluation_seconds'] >= 0
    assert history.config['wpo_converter']['f_lst'] == [1.]
    methods = ['baseline', 'hat', 'wpo'] if hat_epochs else ['baseline', 'wpo']
    assert [r['method'] for r in history.results] == methods
    assert all(r['realization_id'] == 'test1' and r['count'] == 2 and
               np.isfinite(r['loss']) for r in history.results)
    assert json.loads(results.read_text()) == asdict(history)
    assert set(tmp_path.iterdir()) == {checkpoint, results}
    for name, value in torch.load(checkpoint, weights_only=True).items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)


def test_workers_default_to_slurm_allocation(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '32')
    assert wpo.parse_args([]).workers == 32
    assert wpo.parse_args(['--dry-run']).workers == 1
    assert wpo.parse_args(['--dry-run', '--workers', '2']).workers == 2


def test_callback_timing_rng_serialization_and_stock_stopping(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(wpo, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(wpo, 'candidate_converter', lambda optimizer, x: x.copy())
    optimizer = SimpleNamespace(loss_margin=.125, loss_baseline=2.)
    evaluations, saves = [], []
    def evaluate(converter):
        evaluations.append(converter)
        random.random()
        np.random.random()
        torch.rand(1)
        clock[0] += 100.
        return [dict(loss=.2, accuracy=.5, count=2, realization_id='validation1', t_inference=86400.)]
    callback = wpo.WPOHistoryCallback(optimizer, evaluate, 10., on_progress=lambda rows: saves.append(len(rows)))
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    candidate = SimpleNamespace(x=np.array([.5]), fun=.5)
    clock[0] = 1.
    assert callback(candidate) is False
    assert callback.history[0].metrics['tnmse'] == 2.75
    clock[0] += 2.
    callback(candidate)
    assert len(evaluations) == 1
    clock[0] += 10.
    candidate.fun = -.1
    with pytest.raises(StopIteration):
        callback(candidate)
    assert len(evaluations) == 2
    assert [r.seconds for r in callback.history if r.realization_id is None] == [1., 3., 13.]
    assert callback.evaluation_seconds == 200.
    assert saves == [2, 3, 5]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state
    # WPO deep-copies and pickles callbacks in each worker's objective payload.
    assert pickle.loads(pickle.dumps(callback)).__dict__ == {}


def test_stock_wpo_with_spawned_workers_and_progress_history():
    args = wpo.parse_args(['--dry-run', '--workers', '2', '--optimize-significance'])
    cfg = common.make_rpu_config([1.])
    cfg.forward.is_perfect = True
    analog = convert_to_analog(nn.Linear(3, 2), cfg).eval()
    before = [tuple(value.clone() for value in tile.get_weights()) for tile in analog.analog_tiles()]
    saves = []
    threads = torch.get_num_threads()
    environment = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    converter, history = wpo.optimize_wpo(args, analog, None, on_progress=lambda rows: saves.append(len(rows)))
    assert torch.get_num_threads() == threads
    assert {key: os.environ.get(key) for key in environment} == environment
    assert len(converter.g_lst) == 2 and len(converter.f_lst) == 1
    assert 0.1 <= converter.f_lst[0] <= 5.
    assert all(np.isfinite(r.metrics['tnmse']) for r in history)
    assert saves and saves[-1] == len(history)
    for tile, initial in zip(analog.analog_tiles(), before):
        for value, expected in zip(tile.get_weights(), initial):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)


def test_weight_distribution_is_normalized_per_tile():
    cfg = common.make_rpu_config([1.])
    cfg.mapping.weight_scaling_omega = 0.
    analog = convert_to_analog(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), cfg)
    tiles = list(analog.analog_tiles())
    tiles[0].set_weights(torch.tensor([[2., -1.], [0., .5]]), torch.zeros(2))
    tiles[1].set_weights(torch.tensor([[20., -10.], [0., 5.]]), torch.zeros(2))
    weights = wpo.collect_wpo_weights(analog)
    torch.testing.assert_close(weights[:4], weights[4:])
    assert weights.abs().max() == 1.


def test_module_command_with_real_workers_and_offline_mnist(tmp_path):
    # Tiny, valid IDX files exercise torchvision and `python -m` without downloads.
    data_dir = tmp_path / 'data'
    raw = data_dir / 'MNIST' / 'raw'
    raw.mkdir(parents=True)
    for split in ['train', 't10k']:
        (raw / f'{split}-images-idx3-ubyte').write_bytes(
            struct.pack('>IIII', 2051, 2, 28, 28) + bytes(2 * 28 * 28),
        )
        (raw / f'{split}-labels-idx1-ubyte').write_bytes(struct.pack('>II', 2049, 2) + bytes([0, 1]))
    checkpoint, results = tmp_path / 'digital.pt', tmp_path / 'wpo.json'
    torch.save(common.Net().state_dict(), checkpoint)
    completed = subprocess.run([
        sys.executable, '-m', 'experiments.mnist.wpo', '--dry-run', '--workers', '2',
        '--no-accel', '--checkpoint', str(checkpoint), '--results-file', str(results),
        '--data-dir', str(data_dir), '--batch-size', '2', '--test-batch-size', '2',
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
       env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
       timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "has overridden updating='immediate'" not in completed.stderr
    history = json.loads(results.read_text())
    assert history['config']['stage'] == 'complete'
    assert history['config']['workers'] == 2
    assert [row['method'] for row in history['results']] == ['baseline', 'wpo']
    assert any(row['realization_id'] == 'validation1' for row in history['optimization_history'])
