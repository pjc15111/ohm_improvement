"""Train and save the shared MNIST digital baseline from the command line."""

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from experiments.common import evaluate_classifier
from experiments.history import ExperimentHistory, HistoryRecord
from experiments.mnist.common import (
    Net, log_commit, make_loaders, report_metrics, synchronize_device,
)


@dataclass
class DigitalConfig:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 14
    lr: float = 1.0
    gamma: float = 0.7
    seed: int = 1
    log_interval: int = 10
    data_dir: str = 'data'
    checkpoint: str = 'results/mnist/digital.pt'
    results_file: str = 'results/mnist/digital.json'
    no_accel: bool = False
    dry_run: bool = False


def parse_args(argv=None) -> DigitalConfig:
    defaults = DigitalConfig()
    parser = argparse.ArgumentParser(description='Train the shared MNIST digital baseline')
    parser.add_argument('--batch-size', type=int, default=defaults.batch_size)
    parser.add_argument('--test-batch-size', type=int, default=defaults.test_batch_size)
    parser.add_argument('--epochs', type=int, default=defaults.epochs)
    parser.add_argument('--lr', type=float, default=defaults.lr)
    parser.add_argument('--gamma', type=float, default=defaults.gamma)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--log-interval', type=int, default=defaults.log_interval)
    parser.add_argument('--data-dir', default=defaults.data_dir)
    parser.add_argument('--checkpoint', default=defaults.checkpoint)
    parser.add_argument('--results-file', default=defaults.results_file)
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='one epoch with one training and test batch')
    config = DigitalConfig(**vars(parser.parse_args(argv)))
    if any(value <= 0 for value in (config.batch_size, config.test_batch_size,
                                   config.epochs, config.log_interval)):
        parser.error('batch sizes, epochs, and log interval must be positive')
    if any(not math.isfinite(value) or value <= 0 for value in (config.lr, config.gamma)):
        parser.error('learning rate and gamma must be finite and positive')
    if not 0 <= config.seed < 2**63:
        parser.error('seed must be an integer in [0, 2**63)')
    if config.dry_run:
        config.epochs = 1
    return config


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
        batches.append(HistoryRecord(
            batch=batch_index + 1, seconds=perf_counter() - start,
            metrics=dict(loss=loss.item(), accuracy=batch_correct / targets.numel()), count=targets.numel(),
        ))
        if batch_index % log_interval == 0:
            print(f"Digital batch {batch_index}/{len(loader)}: loss={loss.item():.4f}")
    return dict(loss=total_loss / count, accuracy=correct / count, count=count, batches=batches)


def train_digital_model(config: DigitalConfig, train_loader, test_loader, device):
    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.gamma)
    history = []
    start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        offset = perf_counter() - start
        training = train_digital_epoch(model, train_loader, device, optimizer, config.log_interval)
        for batch in training['batches']:
            batch.step = (epoch - 1) * len(train_loader) + batch.batch
            batch.epoch = epoch - 1 + batch.batch / len(train_loader)
            batch.seconds += offset
        evaluation = evaluate_classifier(model, test_loader, device)
        report_metrics(f"Digital epoch {epoch} training ({perf_counter() - start:.2f}s)", training)
        report_metrics(f"Digital epoch {epoch} evaluation", evaluation)
        history.append(dict(epoch=epoch, seconds=perf_counter() - start,
                            step=epoch * len(train_loader), training=training, evaluation=evaluation))
        scheduler.step()
    return model.eval(), history


def main(argv=None):
    config = parse_args(argv)
    log_commit()
    start = perf_counter()
    torch.manual_seed(config.seed)
    device = torch.device('cuda' if not config.no_accel and torch.cuda.is_available() else 'cpu')
    print(f'Device: {torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}', flush=True)
    train_loader, test_loader = make_loaders(config, device)
    model, digital_history = train_digital_model(config, train_loader, test_loader, device)
    checkpoint = Path(config.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    history = ExperimentHistory(
        method='digital', config=asdict(config), digital_history=digital_history,
        digital_baseline=digital_history[-1]['evaluation'],
    )
    history.save(config.results_file)
    print(f'Checkpoint: {checkpoint}')
    print(f'Results: {config.results_file}')
    print(f'Total runtime: {perf_counter() - start:.2f}s')
    return history


if __name__ == '__main__':
    main()
