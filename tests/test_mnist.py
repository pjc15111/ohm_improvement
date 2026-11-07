from dataclasses import asdict, is_dataclass
from itertools import groupby
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from experiments.mnist import common as mnist_common, digital as mnist_digital, oi as mnist


def fake_mnist(root, train, download, transform):
    generator = torch.Generator().manual_seed(12 if train else 13)
    return TensorDataset(torch.randn(4, 1, 28, 28, generator=generator),
                         torch.tensor([0, 1, 2, 3]))


def test_loaders_shuffle_only_training_and_bound_dry_run(monkeypatch):
    monkeypatch.setattr(mnist_common.datasets, 'MNIST', fake_mnist)
    args = mnist.parse_args(['--dry-run', '--batch-size', '2', '--test-batch-size', '3'])
    train, test = mnist.make_loaders(args, torch.device('cpu'))
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert len(train.dataset) == 2 and len(test.dataset) == 3
    assert len(train) == len(test) == 1
    assert args.oi_epochs == 1
    assert args.train_realizations == args.evaluation_realizations == 1
    assert len(args.times) == 1


def test_oi_batch_updates_only_selected_parameters():
    # A small inference model exercises the batch update without MNIST data.
    from aihwkit.nn.conversion import convert_to_analog
    from ohm_improvement.model import OIModel
    from ohm_improvement.tiles import OITile
    config = mnist.make_rpu_config([1., 2.])
    config.forward.is_perfect = True
    analog = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(dim=1)),
                               rpu_config=config)
    model = OIModel(analog)
    tile = next(t for t in model.modules() if isinstance(t, OITile))
    before = [g.detach().clone() for g in tile.g_plus]
    bias = tile.bias.detach().clone()
    significance = tile.significance.detach().clone()
    optimizer = torch.optim.Adam(model.optimization_parameters(include_output_scaling=False, include_bias=False, include_significance=False, conductance_lr=.01))
    metrics = mnist.optimize_oi_batch(model.eval(), torch.tensor([[.2, -.4, .8], [.1, .3, -.2]]),
                                      torch.tensor([0, 1]), mnist.training_samples(
                                          ['train1', 'train2'], [0., 86400.], step=1,
                                          noise_mode='fixed', time_sampling='all', generator=None), optimizer)
    assert metrics['count'] == 8
    assert len(metrics['samples']) == 4
    assert {(r.realization_id, r.time_index) for r in metrics['samples']} == {
        ('train1/time0', 0), ('train1/time1', 1), ('train2/time0', 0), ('train2/time1', 1)}
    assert abs(sum(r.metrics['loss'] * r.count for r in metrics['samples']) - metrics['loss_sum']) < 1e-6
    assert any(not torch.equal(a, b) for a, b in zip(before, tile.g_plus))
    torch.testing.assert_close(tile.bias, bias)
    torch.testing.assert_close(tile.significance, significance)
    assert all(torch.isfinite(g).all() for g in tile.g_plus)


def test_mnist_synthetic_dry_run(monkeypatch, tmp_path):
    monkeypatch.setattr(mnist_common.datasets, 'MNIST', fake_mnist)
    checkpoint = tmp_path / 'digital.pt'
    digital_results = tmp_path / 'digital.json'
    digital = mnist_digital.main([
        '--dry-run', '--no-accel', '--batch-size', '2', '--test-batch-size', '2',
        '--checkpoint', str(checkpoint), '--results-file', str(digital_results),
    ])
    assert digital.method == 'digital'
    assert len(digital.digital_history) == 1
    weights = torch.load(checkpoint, weights_only=True)
    assert json.loads(digital_results.read_text()) == asdict(digital)
    results = mnist.main(['--dry-run', '--no-accel', '--batch-size', '2', '--test-batch-size', '2',
                          '--checkpoint', str(checkpoint),
                          '--results-file', str(tmp_path / 'results.json')])
    assert {r['data_split'] for r in digital.results} == {'seen', 'unseen'}
    assert results.method == 'oi'
    assert results.digital_history == []
    assert len(results.optimization_history) == 1
    assert results.digital_baseline == digital.digital_baseline
    assert [r['method'] for r in results.results] == ['baseline', 'baseline', 'oi', 'oi']
    assert all(r['realization_id'] == 'test1' and r['count'] == 2
               and torch.isfinite(torch.tensor(r['loss'])) for r in results.results)

    saved = json.loads((tmp_path / 'results.json').read_text())
    assert is_dataclass(results)
    assert saved == asdict(results)
    assert saved['digital_history'] == []
    sample = saved['optimization_history'][0]['samples'][0]
    assert set(sample['metrics']) == {'loss', 'accuracy'}
    assert sample['step'] == 1 and sample['epoch'] == 1
    assert sample['seconds'] >= 0 and sample['realization_id'] == 'train1/time0/step1'
    assert sample['t_inference'] == saved['config']['times'][0]
    assert {(r['data_split'], r['realization_split']) for r in saved['diagnostics']} == {
        ('seen', 'seen'), ('seen', 'unseen'), ('unseen', 'seen'), ('unseen', 'unseen')}
    assert {r['realization_id'] for r in saved['diagnostics'] if r['realization_split'] == 'seen'} == {
        sample['realization_id']}
    reloaded = torch.load(checkpoint, weights_only=True)
    for name, value in weights.items():
        torch.testing.assert_close(reloaded[name], value, rtol=0, atol=0)
    assert set(tmp_path.iterdir()) == {checkpoint, digital_results, tmp_path / 'results.json'}


def test_history_steps_continue_across_epochs_and_retain_all_samples(monkeypatch):
    from aihwkit.nn.conversion import convert_to_analog
    monkeypatch.setattr(mnist_digital, 'Net', lambda: nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(dim=1)))
    args = mnist.parse_args(['--oi-epochs', '2', '--train-realizations', '2', '--noise-mode', 'fresh',
                             '--times', '0', '86400'])
    loader = DataLoader(TensorDataset(torch.tensor([[.2, -.4, .8], [.1, .3, -.2], [.4, .2, .1]]),
                                     torch.tensor([0, 1, 0])), batch_size=2)
    device = torch.device('cpu')
    digital, history = mnist_digital.train_digital_model(
        mnist_digital.DigitalConfig(epochs=2), loader, loader, device,
    )
    batches = [batch for epoch in history for batch in epoch['training']['batches']]
    assert [b.step for b in batches] == [1, 2, 3, 4]
    assert [b.epoch for b in batches] == [.5, 1., 1.5, 2.]
    assert [b.seconds for b in batches] == sorted(b.seconds for b in batches)
    config = mnist.make_rpu_config([1.])
    config.forward.is_perfect = True
    analog = convert_to_analog(digital, rpu_config=config)
    _, history = mnist.optimize_oi(args, analog, loader, device)
    samples = [sample for epoch in history for sample in epoch['samples']]
    assert len(samples) == 16
    assert len({r.realization_id for r in samples}) == 16
    assert all(is_dataclass(sample) for sample in samples)
    steps = []
    for step, records in groupby(samples, key=lambda sample: sample.step):
        records = list(records)
        steps.append(dict(step=step, count=sum(r.count for r in records), seconds=records[0].seconds))
    assert [r['step'] for r in steps] == [1, 2, 3, 4]
    assert [r['count'] for r in steps] == [8, 4, 8, 4]
    assert [r['seconds'] for r in steps] == sorted(r['seconds'] for r in steps)
    for epoch in history:
        records = epoch['samples']
        assert abs(sum(r.metrics['loss'] * r.count for r in records) / epoch['count'] - epoch['loss']) < 1e-6


def test_slurm_cpu_threads_are_explicit_and_do_not_affect_gpu(monkeypatch):
    calls = []
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '64')
    monkeypatch.setattr(torch, 'set_num_threads', calls.append)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 64)
    mnist_common.configure_cpu_threads(torch.device('cpu'))
    mnist_common.configure_cpu_threads(torch.device('cuda'))
    assert calls == [64]
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    mnist_common.configure_cpu_threads(torch.device('cpu'))
    assert calls == [64]
