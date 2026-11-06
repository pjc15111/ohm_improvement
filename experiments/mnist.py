"""MNIST training and programming comparison (adapted from PyTorch's example)."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from functools import partial
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import NPairConductanceConverter
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import InferenceRPUConfig

import ohm_improvement as oi
from experiments.common import (
    evaluate_classifier, evaluate_programming, evaluator_from_analog, evaluator_from_oi,
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 64)
        self.l2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.log_softmax(x, dim=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MNIST programming comparison")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='one batch/epoch and one noise/time sample')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--output-dir', help='history/plot directory (default: timestamped results/mnist run)')
    parser.add_argument('--plot-x', choices=['seconds', 'step', 'epoch'], default='seconds')
    parser.add_argument('--train-noise-seed', type=int, default=101)
    parser.add_argument('--validation-noise-seed', type=int, default=202)
    parser.add_argument('--evaluation-noise-seed', type=int, default=303)
    parser.add_argument('--oi-epochs', type=int, default=1)
    parser.add_argument('--oi-lr', '--conductance-lr', dest='oi_lr', type=float, default=0.01,
                        help='OI conductance learning rate')
    parser.add_argument('--significance-lr', type=float, default=0.0001)
    parser.add_argument('--bias-lr', type=float, default=0.0001)
    parser.add_argument('--output-scaling-lr', type=float, default=0.0001)
    parser.add_argument('--train-realizations', type=int, default=4)
    parser.add_argument('--evaluation-realizations', type=int, default=4)
    parser.add_argument('--times', type=float, nargs='+', default=[86400, 30 * 86400, 365 * 86400])
    parser.add_argument('--pair-factors', type=float, nargs='+', default=[1.0])
    parser.add_argument('--optimize-significance', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    seeds = (args.train_noise_seed, args.validation_noise_seed, args.evaluation_noise_seed)
    if len(set(seeds)) != 3 or any(not 0 <= seed < 2**63 for seed in seeds):
        parser.error('noise seeds must be distinct integers in [0, 2**63)')
    if any(value <= 0 for value in (args.batch_size, args.test_batch_size, args.epochs,
                                   args.oi_epochs, args.train_realizations,
                                   args.evaluation_realizations, args.log_interval)):
        parser.error('batch sizes, epochs, realization counts, and log interval must be positive')
    if any(not 0 <= t < float('inf') for t in args.times):
        parser.error('times must be finite and nonnegative')
    if any(not 0 < f < float('inf') for f in args.pair_factors):
        parser.error('pair factors must be finite and positive')
    if args.dry_run:
        args.epochs = args.oi_epochs = 1
        args.train_realizations = args.evaluation_realizations = 1
        args.times = args.times[:1]
    return args


def select_device(no_accel=False):
    if not no_accel and torch.cuda.is_available() and rpu_base.cuda.is_compiled():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def make_loaders(args, device):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
    if args.dry_run:
        train_data = Subset(train_data, range(min(args.batch_size, len(train_data))))
        test_data = Subset(test_data, range(min(args.test_batch_size, len(test_data))))
    workers = dict(num_workers=1, persistent_workers=True, pin_memory=True) if device.type == 'cuda' else {}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **workers)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **workers)
    return train_loader, test_loader


def make_rpu_config(pair_factors):
    config = InferenceRPUConfig()
    config.noise_model = PCMLikeNoiseModel(
        g_converter=NPairConductanceConverter(list(pair_factors), g_min=0., g_max=25.),
    )
    config.drift_compensation = GlobalDriftCompensation()
    config.mapping.max_input_size = 0
    config.mapping.max_output_size = 0
    config.forward.is_perfect = False
    return config


def report_metrics(label, metrics):
    print(f"{label}: loss={metrics['loss']:.4f}, accuracy={100 * metrics['accuracy']:.2f}%")


def synchronize_device(device):
    """Include completed GPU work in history timestamps."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_digital_epoch(model, loader, device, optimizer, log_interval):
    model.train()
    synchronize_device(device)
    start = perf_counter()
    batches = []
    total_loss, correct, count = 0., 0, 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.numel()
        batch_correct = (output.argmax(dim=-1) == targets).sum().item()
        correct += batch_correct
        count += targets.numel()
        synchronize_device(device)
        batches.append(dict(batch=batch_index + 1, seconds=perf_counter() - start,
                            loss=loss.item(), accuracy=batch_correct / targets.numel(),
                            count=targets.numel()))
        if batch_index % log_interval == 0:
            print(f"Digital batch {batch_index}/{len(loader)}: loss={loss.item():.4f}")
    return dict(loss=total_loss / count, accuracy=correct / count, count=count, batches=batches)


def train_digital_model(args, train_loader, test_loader, device):
    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    history = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        offset = perf_counter() - start
        training = train_digital_epoch(model, train_loader, device, optimizer, args.log_interval)
        for batch in training['batches']:
            batch.update(step=(epoch - 1) * len(train_loader) + batch['batch'],
                         epoch=epoch - 1 + batch['batch'] / len(train_loader),
                         seconds=offset + batch['seconds'])
        evaluation = evaluate_classifier(model, test_loader, device)
        report_metrics(f"Digital epoch {epoch} training", training)
        report_metrics(f"Digital epoch {epoch} evaluation", evaluation)
        history.append(dict(epoch=epoch, seconds=perf_counter() - start,
                            step=epoch * len(train_loader), training=training, evaluation=evaluation))
        scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), 'mnist_nn.pt')
    return model.eval(), history


def optimize_oi_batch(oi_model, inputs, targets, realizations, n_times, optimizer):
    """Average task loss over hardware samples, then update and clamp once."""
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct = 0., 0
    samples = []
    n_samples = len(realizations) * n_times
    for realization in realizations:
        for time_index in range(n_times):
            oi.program_weights(oi_model, realization, time_index)
            output = oi_model(inputs)
            loss = F.nll_loss(output, targets)
            (loss / n_samples).backward()
            total_loss += loss.item() * targets.numel()
            sample_correct = (output.argmax(dim=-1) == targets).sum().item()
            correct += sample_correct
            samples.append(dict(realization_id=f'train{realization + 1}', time_index=time_index,
                                loss=loss.item(), accuracy=sample_correct / targets.numel(),
                                count=targets.numel()))
    optimizer.step()
    oi.clamp_model(oi_model)
    return dict(loss_sum=total_loss, correct=correct, count=targets.numel() * n_samples, samples=samples)


def optimize_oi_epoch(oi_model, loader, device, realizations, times, optimizer, log_interval):
    # Keep backend inference behavior while optimizing conductances with autograd.
    oi_model.eval()
    synchronize_device(device)
    start = perf_counter()
    samples = []
    total_loss, correct, count = 0., 0, 0
    for batch_index, (inputs, targets) in enumerate(loader):
        batch = optimize_oi_batch(oi_model, inputs.to(device), targets.to(device),
                                  realizations, len(times), optimizer)
        synchronize_device(device)
        elapsed = perf_counter() - start
        samples.extend(dict(record, batch=batch_index + 1, seconds=elapsed,
                            t_inference=float(times[record['time_index']]))
                       for record in batch['samples'])
        total_loss += batch['loss_sum']
        correct += batch['correct']
        count += batch['count']
        if batch_index % log_interval == 0:
            report_metrics(f"OI batch {batch_index}/{len(loader)}",
                           dict(loss=total_loss / count, accuracy=correct / count))
    return dict(loss=total_loss / count, accuracy=correct / count, count=count, samples=samples)


def optimize_oi(args, analog_model, train_loader, device):
    oi_model = oi.convert_to_oi(
        analog_model, n_realizations=args.train_realizations, t_inferences=args.times,
        noise_seeds=dict(train=args.train_noise_seed, validation=args.validation_noise_seed,
                         evaluation=args.evaluation_noise_seed),
    )
    parameters = oi.optimization_parameters(
        oi_model, include_conductances=True, include_output_scaling=False,
        include_bias=False, include_significance=args.optimize_significance,
        conductance_lr=args.oi_lr, significance_lr=args.significance_lr,
        bias_lr=args.bias_lr, output_scaling_lr=args.output_scaling_lr,
    )
    optimizer = torch.optim.Adam(parameters)
    history = []
    start = perf_counter()
    for epoch in range(1, args.oi_epochs + 1):
        offset = perf_counter() - start
        metrics = optimize_oi_epoch(oi_model, train_loader, device, range(args.train_realizations),
                                    args.times, optimizer, args.log_interval)
        elapsed = perf_counter() - start
        for sample in metrics['samples']:
            sample.update(step=(epoch - 1) * len(train_loader) + sample['batch'],
                          epoch=epoch - 1 + sample['batch'] / len(train_loader),
                          seconds=offset + sample['seconds'])
        history.append(dict(epoch=epoch, seconds=elapsed, **metrics))
        report_metrics(f"OI epoch {epoch} ({elapsed:.2f}s)", metrics)
    return oi_model, history


def run_comparison(analog_model, oi_model, loader, device, realization_ids, times, seed):
    """Evaluate ordinary and optimized mappings through the same stock simulator."""
    evaluate = partial(evaluate_classifier, loader=loader, device=device)
    # Snapshot both schemes before evaluation mutates the retained analog model.
    evaluators = {
        'baseline': evaluator_from_analog(analog_model, seed=seed),
        'oi': evaluator_from_oi(oi_model, analog_model, seed=seed),
    }
    realization_ids = tuple(realization_ids)
    times = tuple(times)
    results = []
    for method, evaluator in evaluators.items():
        for record in evaluate_programming(evaluator, evaluate, realization_ids, times):
            results.append(dict(method=method, **record))
            report_metrics(f"{method} {record['realization_id']} t={record['t_inference']}s", record)
    return results



def save_run(history, output_dir=None, plot_x='seconds'):
    """Save raw histories before plotting so the figures can be regenerated."""
    directory = Path(output_dir) if output_dir else Path('results/mnist') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'history.json'
    path.write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
    from experiments.plotting import plot_histories
    plot_histories(history, directory, x_axis=plot_x)
    print(f'History and plots saved to {directory}')
    return directory


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = select_device(args.no_accel)
    train_loader, test_loader = make_loaders(args, device)
    model, digital_history = train_digital_model(args, train_loader, test_loader, device)
    analog_model = convert_to_analog(model, rpu_config=make_rpu_config(args.pair_factors)).to(device)
    oi_model, oi_history = optimize_oi(args, analog_model, train_loader, device)
    results = run_comparison(
        analog_model, oi_model, test_loader, device,
        [f'test{i + 1}' for i in range(args.evaluation_realizations)],
        args.times, args.evaluation_noise_seed,
    )
    history = dict(schema_version=1, config=vars(args),
                   digital_baseline=digital_history[-1]['evaluation'],
                   digital_history=digital_history, oi_history=oi_history, results=results)
    save_run(history, args.output_dir, args.plot_x)
    return history


if __name__ == '__main__':
    main()
