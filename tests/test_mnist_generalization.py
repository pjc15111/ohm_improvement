"""Exact seen-noise diagnostics and shared HAT initialization."""

from dataclasses import asdict
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator import rpu_base

from experiments.common import evaluate_classifier
from experiments.mnist import common, hat, oi, wpo
from ohm_improvement import OIModel


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and (not torch.cuda.is_available() or not rpu_base.cuda.is_compiled()):
        pytest.skip('CUDA-capable PyTorch and AIHWKIT required')
    return torch.device(request.param)


def test_cartesian_diagnostic_preserves_exact_programmed_noise(device):
    config = common.make_rpu_config([1.])
    config.forward.is_perfect = True
    model = OIModel(convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(-1)), config).to(device), seed=17)
    x = torch.tensor([[.2, -.4, .8], [.1, .3, -.2], [.4, .2, .1]])
    labels = torch.tensor([0, 1, 0])
    loaders = {
        'seen': DataLoader(TensorDataset(x, labels), batch_size=2),
        'unseen': DataLoader(TensorDataset(-x[:2], labels[:2]), batch_size=2),
    }
    optimizer = torch.optim.Adam(model.optimization_parameters())
    oi.optimize_oi_batch(model.eval(), x.to(device), labels.to(device), [('train1', 0, 0.), ('train2', 1, 86400.)], optimizer)
    results = oi.evaluate_generalization(model, loaders,
                                         {'seen': ['train1', 'train2'], 'unseen': ['test1', 'test2']},
                                         [0., 86400., 172800.], device, training_times=[0., 86400.])
    assert len(results) == 24
    assert {r['time_split'] for r in results} == {'seen', 'unseen'}
    assert all(r['time_split'] == ('unseen' if r['t_inference'] == 172800. else 'seen') for r in results)
    assert {(r['data_split'], r['realization_split']) for r in results} == {
        ('seen', 'seen'), ('seen', 'unseen'), ('unseen', 'seen'), ('unseen', 'unseen')}
    # Rebuild exactly the same noise in OI, and compare against its stock snapshot.
    for record in results:
        model.program(record['realization_id'])
        model.drift(record['t_inference'])
        expected = evaluate_classifier(model.eval(), loaders[record['data_split']], device)
        assert record['count'] == (3 if record['data_split'] == 'seen' else 2)
        assert record['accuracy'] == expected['accuracy']
        assert record['loss'] == pytest.approx(expected['loss'], rel=3e-5, abs=3e-6)


@pytest.mark.parametrize('noise_mode', ['fixed', 'fresh'])
def test_noise_schedule_and_seen_realization_selection(noise_mode):
    args = oi.parse_args(['--oi-epochs', '2', '--train-realizations', '2', '--noise-mode', noise_mode, '--times', '0'])
    model = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(-1)), common.make_rpu_config([1.]))
    loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)
    _, history = oi.optimize_oi(args, model, loader, torch.device('cpu'))
    ids = [r.realization_id for epoch in history for r in epoch['samples']]
    assert len(set(ids)) == (2 if noise_mode == 'fixed' else 8)
    assert oi.seen_realizations(history, 2) == (['train1/time0', 'train2/time0'] if noise_mode == 'fixed' else
                                               ['train1/time0/step1', 'train1/time0/step3'])
    assert set(oi.seen_realizations(history, 16)) == set(ids)


def test_shared_hat_checkpoint_initializes_both_oi_conditions_and_wpo(monkeypatch, tmp_path, device):
    def fake_mnist(root, train, download, transform):
        generator = torch.Generator().manual_seed(12 if train else 13)
        return TensorDataset(torch.randn(4, 1, 28, 28, generator=generator), torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr(common.datasets, 'MNIST', fake_mnist)
    digital_path, hat_path = tmp_path / 'digital.pt', tmp_path / 'hat.pt'
    torch.save(common.Net().state_dict(), digital_path)
    hat_result = hat.main([
        '--checkpoint', str(digital_path), '--output-checkpoint', str(hat_path),
        '--results-file', str(tmp_path / 'hat.json'), '--hat-epochs', '1',
        '--batch-size', '4', '--test-batch-size', '4', '--no-accel',
    ])
    assert hat_result.method == 'hat' and len(hat_result.optimization_history) == 1
    checkpoint_bytes = hat_path.read_bytes()
    assert json.loads((tmp_path / 'hat.json').read_text()) == asdict(hat_result)
    initial_weights = []
    original_oi, original_wpo = oi.optimize_oi, wpo.optimize_wpo
    def capture_weights(model):
        values = []
        for tile in model.analog_tiles():
            weights, bias = tile.get_weights(apply_weight_scaling=False)
            values.extend(tensor.detach().cpu().clone() for tensor in
                          [weights, bias, tile.get_mapping_scales(), tile.out_scaling_alpha]
                          if tensor is not None)
        initial_weights.append(values)
    def capture_oi(args, model, *positional, **keywords):
        capture_weights(model)
        return original_oi(args, model, *positional, **keywords)
    def capture_wpo(args, model, *positional, **keywords):
        capture_weights(model)
        return original_wpo(args, model, *positional, **keywords)
    monkeypatch.setattr(oi, 'optimize_oi', capture_oi)
    monkeypatch.setattr(wpo, 'optimize_wpo', capture_wpo)
    for count, sampling in [(4, 'all'), (1, 'random')]:
        result = oi.main([
            '--checkpoint', str(digital_path), '--hat-checkpoint', str(hat_path),
            '--results-file', str(tmp_path / f'oi-{count}.json'), '--oi-epochs', '1',
            '--train-realizations', str(count), '--time-sampling', sampling, '--evaluation-realizations', '1',
            '--batch-size', '4', '--test-batch-size', '3',
            *(['--no-accel'] if device.type == 'cpu' else []),
        ])
        assert result.config['hat_checkpoint'] == str(hat_path)
        samples_per_step = count * 3 if sampling == 'all' else count
        assert result.optimization_history[0]['count'] == 4 * samples_per_step
        assert len(result.optimization_history[0]['samples']) == samples_per_step
        assert {r['time_split'] for r in result.results} == {'seen', 'unseen'}
        assert {r['time_split'] for r in result.diagnostics} == {'seen', 'unseen'}
        assert {r['method'] for r in result.results} == {'baseline', 'oi', 'hat'}
        assert {(r['data_split'], r['count']) for r in result.diagnostics} == {('seen', 4), ('unseen', 4)}
    wpo_result = wpo.main([
        '--checkpoint', str(digital_path), '--hat-checkpoint', str(hat_path),
        '--results-file', str(tmp_path / 'wpo.json'), '--dry-run', '--no-accel',
    ])
    assert wpo_result.config['stage'] == 'complete'
    assert not any(isinstance(r, dict) and r.get('phase') == 'hat' for r in wpo_result.optimization_history)
    assert {r['method'] for r in wpo_result.results} == {'baseline', 'hat', 'wpo'}
    assert hat_path.read_bytes() == checkpoint_bytes
    assert len(initial_weights) == 3
    for weights in initial_weights[1:]:
        assert len(weights) == len(initial_weights[0])
        for actual, expected in zip(weights, initial_weights[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('sampling', ['all', 'random'])
def test_fresh_sampling_is_reproducible_and_does_not_consume_global_rng(sampling):
    def sample(generator, step):
        return oi.training_samples(['train1', 'train2', 'train3', 'train4'], common.TRAINING_TIMES,
                                   step=step, noise_mode='fresh', time_sampling=sampling,
                                   generator=generator)
    before = torch.random.get_rng_state().clone()
    first = sample(torch.Generator().manual_seed(17), 1)
    repeated = sample(torch.Generator().manual_seed(17), 1)
    second = sample(torch.Generator().manual_seed(17), 2)
    assert first == repeated
    assert len(first) == (12 if sampling == 'all' else 4)
    assert len({name for name, _, _ in first}) == len(first)
    assert not {name for name, _, _ in first} & {name for name, _, _ in second}
    assert all(time == common.TRAINING_TIMES[index] for _, index, time in first)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_five_twelve_sample_epochs_and_sixty_single_sample_epochs_have_equal_budgets():
    loader = DataLoader(TensorDataset(torch.tensor([[.2, -.4, .8], [.1, .3, -.2]]),
                                     torch.tensor([0, 1])), batch_size=2)
    counts = []
    for epochs, realizations, sampling in [(5, 4, 'all'), (60, 1, 'random')]:
        args = oi.parse_args(['--oi-epochs', str(epochs), '--train-realizations', str(realizations),
                             '--time-sampling', sampling, '--times', '0', '86400', '2592000'])
        config = common.make_rpu_config([1.])
        config.forward.is_perfect = True
        model = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(-1)), config)
        _, history = oi.optimize_oi(args, model, loader, torch.device('cpu'))
        records = [record for epoch in history for record in epoch['samples']]
        assert len(history) == epochs
        assert len({r.realization_id for r in records}) == len(records)
        assert {r.t_inference for r in records} == set(args.times)
        assert max(r.step for r in records) == epochs
        assert all(len(epoch['samples']) == (12 if sampling == 'all' else 1) for epoch in history)
        counts.append(sum(epoch['count'] for epoch in history))
        for time in args.times:
            ids = oi.seen_realizations(history, 4, time=time)
            used_at_time = {r.realization_id for r in records if r.t_inference == time}
            assert ids and set(ids) <= used_at_time
    assert counts == [120, 120]


def test_wpo_evaluates_held_out_times_without_adding_them_to_training():
    args = wpo.parse_args(['--times', '0', '--unseen-times', '3600', '--evaluation-realizations', '1'])
    config = common.make_rpu_config([1.])
    config.forward.is_perfect = True
    model = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(-1)), config).eval()
    loader = DataLoader(TensorDataset(torch.ones(2, 3), torch.tensor([0, 1])), batch_size=2)
    results = wpo.run_comparison(args, model, config.noise_model.g_converter, loader, seen_loader=loader)
    for method in ['baseline', 'wpo']:
        for time in [0., 3600.]:
            records = [r for r in results if r['method'] == method and r['t_inference'] == time]
            assert {r['data_split'] for r in records} == {'seen', 'unseen'}
            assert records[0]['loss'] == records[1]['loss']
            assert records[0]['accuracy'] == records[1]['accuracy']
    assert args.times == [0.]
    assert {(r['method'], r['t_inference'], r['time_split']) for r in results} == {
        (method, time, split) for method in ['baseline', 'wpo']
        for time, split in [(0., 'seen'), (3600., 'unseen')]}


@pytest.mark.parametrize('train_digital_parameters', [True, False])
def test_oi_can_update_bias_and_output_scaling(train_digital_parameters):
    torch.manual_seed(17)
    config = common.make_rpu_config([1.])
    config.forward.is_perfect = True
    config.mapping.learn_out_scaling = True
    config.mapping.out_scaling_columnwise = True
    config.noise_model.prog_noise_scale = 0.
    config.noise_model.read_noise_scale = 0.
    config.noise_model.drift_scale = 0.
    analog = convert_to_analog(nn.Sequential(nn.Linear(3, 2), nn.LogSoftmax(-1)), config)
    tile = next(analog.analog_tiles())
    bias = tile.bias.detach().clone()
    scaling = tile.out_scaling_alpha.detach().clone()
    args = oi.parse_args(['--oi-epochs', '1', '--train-realizations', '1', '--times', '0',
                         '--bias-lr', '0.001', '--output-scaling-lr', '0.001',
                         *([] if train_digital_parameters else
                           ['--no-optimize-bias', '--no-optimize-output-scaling'])])
    loader = DataLoader(TensorDataset(torch.tensor([[.2, -.4, .8], [.1, .3, -.2]]),
                                     torch.tensor([0, 0])), batch_size=2)
    optimized, _ = oi.optimize_oi(args, analog, loader, torch.device('cpu'))
    optimized_tile = next(optimized.oi_tiles())
    assert torch.equal(optimized_tile.bias, bias) is not train_digital_parameters
    assert torch.equal(optimized_tile.out_scaling_alpha, scaling) is not train_digital_parameters
