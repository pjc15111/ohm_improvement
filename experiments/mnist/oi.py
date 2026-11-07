"""MNIST OI optimization and stock evaluation (adapted from PyTorch's example)."""

import argparse
from copy import deepcopy
from torch.utils.data import DataLoader

from experiments.history import ExperimentHistory, HistoryRecord
from time import perf_counter

import torch
from torch.nn import functional as F

from aihwkit.nn.conversion import convert_to_analog

import ohm_improvement as oi
from experiments.mnist.common import (
    load_digital_model, log_commit, make_loaders, make_rpu_config, report_metrics,
    select_device, synchronize_device,
    TRAINING_TIMES, UNSEEN_TIMES, evaluation_times,
)
from experiments.common import (
    evaluate_classifier, evaluate_programming, evaluator_from_analog, evaluator_from_oi,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MNIST OI optimization and evaluation")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='one batch/epoch and one noise/time sample')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--checkpoint', default='results/mnist/digital.pt')
    parser.add_argument('--hat-checkpoint', help='shared HAT weights to initialize OI')
    parser.add_argument('--results-file', default='results/mnist/oi.json')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--noise-seed', type=int, default=101)
    parser.add_argument('--oi-epochs', type=int, default=1)
    parser.add_argument('--oi-lr', '--conductance-lr', dest='oi_lr', type=float, default=0.01,
                        help='OI conductance learning rate')
    parser.add_argument('--significance-lr', type=float, default=0.0001)
    parser.add_argument('--bias-lr', type=float, default=0.0001)
    parser.add_argument('--output-scaling-lr', type=float, default=0.0001)
    parser.add_argument('--train-realizations', type=int, default=4,
                        help='samples per time in all-time mode, or per update in random-time mode')
    parser.add_argument('--noise-mode', choices=['fixed', 'fresh'], default='fresh',
                        help='reuse a fixed noise bank or draw fresh realizations each batch')
    parser.add_argument('--time-sampling', choices=['all', 'random'], default='all',
                        help='evaluate every training time, or sample one uniformly per realization')
    parser.add_argument('--evaluation-realizations', type=int, default=4)
    parser.add_argument('--times', type=float, nargs='+', default=TRAINING_TIMES)
    parser.add_argument('--unseen-times', type=float, nargs='*', default=UNSEEN_TIMES,
                        help='additional final-evaluation times, excluded from training')
    parser.add_argument('--pair-factors', type=float, nargs='+', default=[1.0])
    parser.add_argument('--optimize-significance', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--optimize-bias', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--optimize-output-scaling', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 0 <= args.noise_seed < 2**63:
        parser.error('noise seed must be an integer in [0, 2**63)')
    if any(value <= 0 for value in (args.batch_size, args.test_batch_size,
                                   args.oi_epochs, args.train_realizations,
                                   args.evaluation_realizations, args.log_interval)):
        parser.error('batch sizes, epochs, realization counts, and log interval must be positive')
    if any(not 0 <= t < float('inf') for t in [*args.times, *args.unseen_times]):
        parser.error('times must be finite and nonnegative')
    if any(not 0 < f < float('inf') for f in args.pair_factors):
        parser.error('pair factors must be finite and positive')
    if args.dry_run:
        args.oi_epochs = 1
        args.train_realizations = args.evaluation_realizations = 1
        args.times = args.times[:1]
        args.unseen_times = []
    return args


def training_samples(realization_ids, times, *, step, noise_mode, time_sampling, generator):
    """Choose independent programming IDs and times without consuming global RNG."""
    samples = []
    for name in realization_ids:
        indices = (range(len(times)) if time_sampling == 'all' else
                   [torch.randint(len(times), (), generator=generator).item()])
        for index in indices:
            realization_id = f'{name}/time{index}'
            if noise_mode == 'fresh':
                realization_id += f'/step{step}'
            samples.append((realization_id, index, float(times[index])))
    return samples


def optimize_oi_batch(oi_model: oi.OIModel, inputs, targets, programming_samples, optimizer):
    """Average task loss over hardware samples, then update and clamp once."""
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct = 0., 0
    samples = []
    n_samples = len(programming_samples)
    for realization_id, time_index, time in programming_samples:
        oi_model.program(realization_id)
        oi_model.drift(time)
        output = oi_model(inputs)
        loss = F.nll_loss(output, targets)
        (loss / n_samples).backward()
        total_loss += loss.item() * targets.numel()
        sample_correct = (output.argmax(dim=-1) == targets).sum().item()
        correct += sample_correct
        samples.append(HistoryRecord(
            realization_id=realization_id, time_index=time_index, t_inference=time, batch=0,
            metrics=dict(loss=loss.item(), accuracy=sample_correct / targets.numel()), count=targets.numel(),
        ))
    optimizer.step()
    oi_model.clamp()
    return dict(loss_sum=total_loss, correct=correct, count=targets.numel() * n_samples, samples=samples)


def optimize_oi_epoch(oi_model: oi.OIModel, loader, device, realization_ids, times, optimizer,
                      log_interval, *, noise_mode='fresh', step_offset=0,
                      time_sampling='all', generator=None):
    # Keep backend inference behavior while optimizing conductances with autograd.
    oi_model.eval()
    synchronize_device(device)
    start = perf_counter()
    samples = []
    total_loss, correct, count = 0., 0, 0
    for batch_index, (inputs, targets) in enumerate(loader):
        programming_samples = training_samples(
            realization_ids, times, step=step_offset + batch_index + 1,
            noise_mode=noise_mode, time_sampling=time_sampling, generator=generator,
        )
        batch = optimize_oi_batch(oi_model, inputs.to(device), targets.to(device),
                                  programming_samples, optimizer)
        synchronize_device(device)
        elapsed = perf_counter() - start
        for record in batch['samples']:
            record.batch = batch_index + 1
            record.seconds = elapsed
        samples.extend(batch['samples'])
        total_loss += batch['loss_sum']
        correct += batch['correct']
        count += batch['count']
        if batch_index % log_interval == 0:
            report_metrics(f"OI batch {batch_index}/{len(loader)}",
                           dict(loss=total_loss / count, accuracy=correct / count))
    return dict(loss=total_loss / count, accuracy=correct / count, count=count, samples=samples)


def optimize_oi(args, analog_model: oi.AnalogModel, train_loader, device):
    oi_model = oi.OIModel(analog_model, seed=args.noise_seed)
    parameters = oi_model.optimization_parameters(
        include_conductances=True, include_output_scaling=args.optimize_output_scaling,
        include_bias=args.optimize_bias, include_significance=args.optimize_significance,
        conductance_lr=args.oi_lr, significance_lr=args.significance_lr,
        bias_lr=args.bias_lr, output_scaling_lr=args.output_scaling_lr,
    )
    optimizer = torch.optim.Adam(parameters)
    samples_per_step = args.train_realizations * (len(args.times) if args.time_sampling == 'all' else 1)
    print(f'OI: noise={args.noise_mode}, samples/update={samples_per_step}, '
          f'epochs={args.oi_epochs}, training times={args.times}, held-out times={args.unseen_times}',
          flush=True)
    history = []
    generator = torch.Generator().manual_seed(args.noise_seed)
    start = perf_counter()
    for epoch in range(1, args.oi_epochs + 1):
        offset = perf_counter() - start
        metrics = optimize_oi_epoch(oi_model, train_loader, device, [f"train{i + 1}" for i in range(args.train_realizations)],
                                    args.times, optimizer, args.log_interval,
                                    noise_mode=args.noise_mode, step_offset=(epoch - 1) * len(train_loader),
                                    time_sampling=args.time_sampling, generator=generator)
        elapsed = perf_counter() - start
        for sample in metrics['samples']:
            sample.step = (epoch - 1) * len(train_loader) + sample.batch
            sample.epoch = epoch - 1 + sample.batch / len(train_loader)
            sample.seconds += offset
        history.append(dict(epoch=epoch, seconds=elapsed, **metrics))
        report_metrics(f"OI epoch {epoch} ({elapsed:.2f}s)", metrics)
    return oi_model, history


def run_comparison(analog_model: oi.AnalogModel, oi_model: oi.OIModel, loader, device,
                   realization_ids, times, seed, *, baseline_model=None, training_times=None,
                   seen_loader=None):
    """Evaluate ordinary and optimized mappings through the same stock simulator."""
    loaders = {'seen': seen_loader, 'unseen': loader} if seen_loader is not None else {'unseen': loader}
    def evaluate(model):
        return {split: evaluate_classifier(model, data, device) for split, data in loaders.items()}
    # Snapshot both schemes before evaluation mutates the retained analog model.
    evaluators = {
        'baseline': evaluator_from_analog(analog_model if baseline_model is None else baseline_model, seed=seed),
        'oi': evaluator_from_oi(oi_model, analog_model, seed=seed),
    }
    if baseline_model is not None:
        evaluators['hat'] = evaluator_from_analog(analog_model, seed=seed)
    realization_ids = tuple(realization_ids)
    times = tuple(times)
    results = []
    for method, evaluator in evaluators.items():
        for record in evaluate_programming(evaluator, evaluate, realization_ids, times):
            for split in loaders:
                result = dict(method=method, data_split=split,
                              **{key: value for key, value in record.items() if key not in loaders},
                              **record[split])
                if training_times is not None:
                    result['time_split'] = 'seen' if record['t_inference'] in training_times else 'unseen'
                results.append(result)
                report_metrics(f"{method} data={split} {record['realization_id']} "
                               f"t={record['t_inference']}s", result)
    return results


def seen_realizations(history, count, *, time=None):
    """Select a deterministic, evenly spaced sample of IDs actually used to train."""
    ids = list(dict.fromkeys(record.realization_id for epoch in history for record in epoch['samples']
                            if time is None or record.t_inference == time))
    count = min(count, len(ids))
    return [ids[index * len(ids) // count] for index in range(count)]


@torch.no_grad()
def evaluate_generalization(oi_model, loaders, realization_groups, times, device, *,
                            training_times=None, seen_by_time=None):
    """Cross data and realization splits using exact OI states in stock snapshots.

    StockProgrammingEvaluator has a different seed derivation, so its `train1`
    would not reproduce OI's training `train1`. Export the actual OI state instead.
    At training times, seen_by_time selects IDs used at that exact time, retaining
    the original read noise as well as programming noise and drift coefficients.
    """
    results = []
    for noise_split, ids in realization_groups.items():
        for time in times:
            selected = seen_by_time.get(time, ids) if noise_split == 'seen' and seen_by_time else ids
            for realization_id in selected:
                oi_model.program(realization_id)
                oi_model.drift(time)
                snapshot = oi_model.to_analog().eval()
                for data_split, loader in loaders.items():
                    metrics = evaluate_classifier(snapshot, loader, device)
                    record = dict(
                        method='oi', data_split=data_split, realization_split=noise_split,
                        realization_id=realization_id, t_inference=float(time),
                        simulator='stock_snapshot', **metrics,
                    )
                    if training_times is not None:
                        record['time_split'] = 'seen' if time in training_times else 'unseen'
                    results.append(record)
                    report_metrics(f'OI data={data_split} noise={noise_split} '
                                   f'{realization_id} t={time}s', metrics)
    return results


def main(argv=None):
    args = parse_args(argv)
    log_commit()
    start = perf_counter()
    torch.manual_seed(args.seed)
    device = select_device(args.no_accel)
    print(f'Device: {torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}', flush=True)
    # Convert every initialization on CPU so paired CPU/GPU runs receive identical
    # normalized tile weights before transferring them to their execution device.
    model = load_digital_model(args.checkpoint, torch.device('cpu'))
    train_loader, test_loader = make_loaders(args, device)
    digital_baseline = evaluate_classifier(model, test_loader, torch.device('cpu'))
    report_metrics('Digital baseline', digital_baseline)
    initial = load_digital_model(args.hat_checkpoint, torch.device('cpu')) if args.hat_checkpoint else model
    analog_model = convert_to_analog(initial, rpu_config=make_rpu_config(args.pair_factors)).to(device)
    baseline_model = (convert_to_analog(model, rpu_config=make_rpu_config(args.pair_factors)).to(device)
                      if args.hat_checkpoint else None)
    oi_model, oi_history = optimize_oi(args, analog_model, train_loader, device)
    times = evaluation_times(args.times, args.unseen_times)
    seen_loader = DataLoader(train_loader.dataset, batch_size=args.test_batch_size, shuffle=False)
    results = run_comparison(
        deepcopy(analog_model), oi_model, test_loader, device,
        [f'test{i + 1}' for i in range(args.evaluation_realizations)],
        times, args.noise_seed, baseline_model=deepcopy(baseline_model), training_times=args.times,
        seen_loader=seen_loader,
    )
    diagnostics = evaluate_generalization(
        oi_model, {'seen': seen_loader, 'unseen': test_loader},
        {'seen': seen_realizations(oi_history, args.evaluation_realizations),
         'unseen': [f'test{i + 1}' for i in range(args.evaluation_realizations)]},
        times, device, training_times=args.times,
        seen_by_time={time: seen_realizations(oi_history, args.evaluation_realizations, time=time)
                      for time in args.times},
    )
    history = ExperimentHistory(
        method='oi', config=vars(args), digital_baseline=digital_baseline,
        optimization_history=oi_history, results=results, diagnostics=diagnostics,
    )
    history.save(args.results_file)
    print(f'Results: {args.results_file}')
    print(f'Total runtime: {perf_counter() - start:.2f}s')
    return history


if __name__ == '__main__':
    main()
