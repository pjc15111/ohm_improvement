"""Model, data, and reporting shared by the MNIST experiments."""

import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import NPairConductanceConverter
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.simulator import rpu_base
from aihwkit.simulator.configs import InferenceRPUConfig


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


def make_loaders(args, device):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.dry_run:
        train_data = Subset(train_data, range(min(args.batch_size, len(train_data))))
        test_data = Subset(test_data, range(min(args.test_batch_size, len(test_data))))
    workers = dict(num_workers=1, persistent_workers=True, pin_memory=True) if device.type == 'cuda' else {}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **workers)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **workers)
    return train_loader, test_loader


def report_metrics(label, metrics):
    print(f"{label}: loss={metrics['loss']:.4f}, accuracy={100 * metrics['accuracy']:.2f}%")


def synchronize_device(device):
    """Include completed GPU work in history timestamps."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def log_commit():
    """Identify the source checkout in ordinary stdout logs."""
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    print(f'Commit: {commit}', flush=True)


def load_digital_model(checkpoint, device):
    """Load the shared baseline without retraining or changing its weights."""
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def select_device(no_accel=False):
    if not no_accel and torch.cuda.is_available() and rpu_base.cuda.is_compiled():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


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
