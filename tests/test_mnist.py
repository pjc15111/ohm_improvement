from dataclasses import asdict, is_dataclass
from itertools import groupby
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from experiments import mnist


def fake_mnist(root, train, download, transform):
    generator = torch.Generator().manual_seed(12 if train else 13)
    return TensorDataset(torch.randn(4, 1, 28, 28, generator=generator),
                         torch.tensor([0, 1, 2, 3]))


def test_loaders_shuffle_only_training_and_bound_dry_run(monkeypatch):
    monkeypatch.setattr(mnist.datasets, 'MNIST', fake_mnist)
    args = mnist.parse_args(['--dry-run', '--batch-size', '2', '--test-batch-size', '3'])
    train, test = mnist.make_loaders(args, torch.device('cpu'))
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert len(train.dataset) == 2 and len(test.dataset) == 3
    assert len(train) == len(test) == 1
    assert args.epochs == args.oi_epochs == 1
    assert args.train_realizations == args.evaluation_realizations == 1
    assert len(args.times) == 1


def test_oi_batch_updates_only_selected_parameters():
    # A small inference model exercises the batch update without MNIST data.
    from aihwkit.nn.conversion import convert_to_analog
    from ohm_improvement.model import convert_to_oi, optimization_parameters
    from ohm_improvement.tiles import OITile
    config = mnist.make_rpu_config([1., 2.])
    config.forward.is_perfect = True
    analog = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(dim=1)),
                               rpu_config=config)
    model = convert_to_oi(analog, 2, [0., 86400.])
    tile = next(t for t in model.modules() if isinstance(t, OITile))
    before = [g.detach().clone() for g in tile.g_plus]
    bias = tile.bias.detach().clone()
    significance = tile.significance.detach().clone()
    optimizer = torch.optim.Adam(optimization_parameters(
        model, include_output_scaling=False, include_bias=False, include_significance=False,
        conductance_lr=.01))
    metrics = mnist.optimize_oi_batch(model.eval(), torch.tensor([[.2, -.4, .8], [.1, .3, -.2]]),
                                      torch.tensor([0, 1]), range(2), 2, optimizer)
    assert metrics['count'] == 8
    assert len(metrics['samples']) == 4
    assert {(r.realization_id, r.time_index) for r in metrics['samples']} == {
        ('train1', 0), ('train1', 1), ('train2', 0), ('train2', 1)}
    assert abs(sum(r.loss * r.count for r in metrics['samples']) - metrics['loss_sum']) < 1e-6
    assert any(not torch.equal(a, b) for a, b in zip(before, tile.g_plus))
    torch.testing.assert_close(tile.bias, bias)
    torch.testing.assert_close(tile.significance, significance)
    assert all(torch.isfinite(g).all() for g in tile.g_plus)


def test_mnist_synthetic_dry_run(monkeypatch, tmp_path):
    monkeypatch.setattr(mnist.datasets, 'MNIST', fake_mnist)
    results = mnist.main(['--dry-run', '--no-accel', '--batch-size', '2', '--test-batch-size', '2',
                          '--results-file', str(tmp_path / 'results.json')])
    assert len(results.digital_history) == len(results.oi_history) == 1
    assert [r['method'] for r in results.results] == ['baseline', 'oi']
    assert all(r['realization_id'] == 'test1' and r['count'] == 2
               and torch.isfinite(torch.tensor(r['loss'])) for r in results.results)

    saved = json.loads((tmp_path / 'results.json').read_text())
    assert is_dataclass(results)
    assert saved == asdict(results)
    assert len(saved['digital_history'][0]['training']['batches']) == 1
    sample = saved['oi_history'][0]['samples'][0]
    assert sample['step'] == 1 and sample['epoch'] == 1
    assert sample['seconds'] >= 0 and sample['realization_id'] == 'train1'
    assert sample['t_inference'] == saved['config']['times'][0]
    assert list(tmp_path.iterdir()) == [tmp_path / 'results.json']


def test_history_steps_continue_across_epochs_and_retain_all_samples(monkeypatch):
    from aihwkit.nn.conversion import convert_to_analog
    monkeypatch.setattr(mnist, 'Net', lambda: nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(dim=1)))
    args = mnist.parse_args(['--epochs', '2', '--oi-epochs', '2', '--train-realizations', '2',
                             '--times', '0', '86400'])
    loader = DataLoader(TensorDataset(torch.tensor([[.2, -.4, .8], [.1, .3, -.2], [.4, .2, .1]]),
                                     torch.tensor([0, 1, 0])), batch_size=2)
    device = torch.device('cpu')
    digital, history = mnist.train_digital_model(args, loader, loader, device)
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
        assert abs(sum(r.loss * r.count for r in records) / epoch['count'] - epoch['loss']) < 1e-6
