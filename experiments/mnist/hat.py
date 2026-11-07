"""Shared MNIST hardware-aware training and checkpoint preparation."""

import argparse
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from aihwkit.nn.conversion import convert_to_analog, convert_to_digital
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.enums import WeightClipType, WeightModifierType

from experiments.common import evaluate_classifier
from experiments.history import ExperimentHistory, HistoryRecord
from experiments.mnist.common import (
    load_digital_model, log_commit, make_loaders, make_rpu_config,
    report_metrics, select_device, synchronize_device,
)
from ohm_improvement import AnalogModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Prepare a shared HAT checkpoint for MNIST')
    parser.add_argument('--checkpoint', default='results/mnist/digital.pt')
    parser.add_argument('--output-checkpoint', default='results/mnist/hat.pt')
    parser.add_argument('--results-file', default='results/mnist/hat.json')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pair-factors', type=float, nargs='+', default=[1.])
    parser.add_argument('--hat-epochs', type=int, default=5)
    parser.add_argument('--hat-lr', type=float, default=.01)
    parser.add_argument('--hat-momentum', type=float, default=.9)
    parser.add_argument('--hat-gamma', type=float, default=.7)
    parser.add_argument('--hat-weight-noise', type=float, default=.1)
    parser.add_argument('--hat-clip-sigma', type=float, default=2.5)
    args = parser.parse_args(argv)
    if args.hat_epochs < 0 or min(args.batch_size, args.test_batch_size, args.log_interval) <= 0:
        parser.error('epochs must be nonnegative; batch sizes and log interval must be positive')
    if any(not 0 < value < float('inf') for value in
           [*args.pair_factors, args.hat_lr, args.hat_gamma, args.hat_clip_sigma]):
        parser.error('pair factors, learning rate, gamma, and clip sigma must be finite and positive')
    if any(not 0 <= value < float('inf') for value in [args.hat_momentum, args.hat_weight_noise]):
        parser.error('momentum and weight noise must be finite and nonnegative')
    if not 0 <= args.seed < 2**63:
        parser.error('seed must be in [0, 2**63)')
    if args.dry_run:
        args.hat_epochs = min(args.hat_epochs, 1)
    return args


def make_hat_config(pair_factors, *, weight_noise=.1, clip_sigma=2.5) -> InferenceRPUConfig:
    """Add Gaussian training weight noise and layerwise Gaussian clipping.

    These are configurable initial settings, not a paper-matched HAT recipe.
    Inference I/O, PCM model, mapping, and compensation match the OI baseline.
    """
    config = make_rpu_config(pair_factors)
    config.modifier.type = WeightModifierType.ADD_NORMAL
    config.modifier.std_dev = weight_noise
    config.modifier.rel_to_actual_wmax = True
    config.modifier.enable_during_test = False
    config.clip.type = WeightClipType.LAYER_GAUSSIAN
    config.clip.sigma = clip_sigma
    return config


def train_hat_epoch(model, loader, device, optimizer, log_interval) -> dict:
    model.train()
    synchronize_device(device)
    start = perf_counter()
    batches = []
    loss_sum, correct, count = 0., 0, 0
    for index, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()  # AnalogSGD also applies the tile's clipping/modifier.
        batch_correct = (output.argmax(-1) == targets).sum().item()
        count += targets.numel()
        correct += batch_correct
        loss_sum += loss.item() * targets.numel()
        synchronize_device(device)
        batches.append(HistoryRecord(
            metrics=dict(loss=loss.item(), accuracy=batch_correct / targets.numel()),
            batch=index, count=targets.numel(), seconds=perf_counter() - start,
        ))
        if (index - 1) % log_interval == 0:
            report_metrics(f'HAT batch {index - 1}/{len(loader)}', batches[-1].metrics)
    return dict(loss=loss_sum / count, accuracy=correct / count, count=count, batches=batches)


def train_hardware_aware(
    args: Namespace, analog_model: AnalogModel, train_loader, validation_loader, device,
) -> tuple[AnalogModel, list[dict]]:
    """Fine-tune a copy of the digital baseline under hardware-aware settings.

    Use AIHWKIT's analog optimizer and retain training/validation history. HAT
    may run on GPU; its resulting weights must move to CPU before stock WPO.
    Preserve the original baseline for the unoptimized comparison.
    """
    model = deepcopy(analog_model)
    if args.hat_epochs == 0:
        return model.cpu().eval(), []
    model.replace_rpu_config(make_hat_config(
        args.pair_factors, weight_noise=args.hat_weight_noise, clip_sigma=args.hat_clip_sigma,
    ))
    model.to(device)
    optimizer = AnalogSGD(model.parameters(), lr=args.hat_lr, momentum=args.hat_momentum)
    optimizer.regroup_param_groups(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.hat_gamma)
    # Activate noise/clipping for the first batch too, before any optimizer step.
    for tile in model.analog_tiles():
        tile.post_update_step()
    history = []
    synchronize_device(device)
    start = perf_counter()
    for epoch in range(1, args.hat_epochs + 1):
        offset = perf_counter() - start
        training = train_hat_epoch(model, train_loader, device, optimizer, args.log_interval)
        for batch in training['batches']:
            batch.step = (epoch - 1) * len(train_loader) + batch.batch
            batch.epoch = epoch - 1 + batch.batch / len(train_loader)
            batch.seconds += offset
        evaluation = evaluate_classifier(model, validation_loader, device)
        elapsed = perf_counter() - start
        history.append(dict(phase='hat', epoch=epoch, step=epoch * len(train_loader),
                            seconds=elapsed, training=training, evaluation=evaluation))
        report_metrics(f'HAT epoch {epoch} training ({elapsed:.2f}s)', training)
        report_metrics(f'HAT epoch {epoch} evaluation', evaluation)
        scheduler.step()
    # Hand WPO clean reference weights, without HAT's modified training copy.
    model.eval()
    model.replace_rpu_config(make_rpu_config(args.pair_factors))
    return model.cpu(), history


def main(argv=None):
    args = parse_args(argv)
    log_commit()
    torch.manual_seed(args.seed)
    device = select_device(args.no_accel)
    print(f'HAT device: {device}', flush=True)
    digital = load_digital_model(args.checkpoint, torch.device('cpu'))
    train_loader, test_loader = make_loaders(args, device)
    baseline = evaluate_classifier(digital, test_loader, torch.device('cpu'))
    analog = convert_to_analog(digital, make_rpu_config(args.pair_factors)).eval()
    trained, history = train_hardware_aware(args, analog, train_loader, test_loader, device)
    checkpoint = Path(args.output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(convert_to_digital(trained).cpu().state_dict(), checkpoint)
    result = ExperimentHistory(method='hat', config=vars(args), digital_baseline=baseline,
                               optimization_history=history)
    result.save(args.results_file)
    print(f'HAT checkpoint: {checkpoint}')
    print(f'Results: {args.results_file}')
    return result


if __name__ == '__main__':
    main()
