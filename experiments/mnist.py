import ohm_improvement as oi

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from time import perf_counter
from copy import deepcopy

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import InferenceRPUConfig

from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.converter.conductance import (
    SinglePairConductanceConverter,
)

# adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py


g_min = 0   # units µS, default for AIHWKIT
g_max = 25  # units µS, default for AIHWKIT


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 64)
        self.l2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_oi_epoch(ensemble, device, train_loader, optimizer, epoch, log_interval):
    ensemble.eval()  # Analog inference behavior; gradients remain enabled.

    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)

        # [realization, time, batch, class]
        outputs = ensemble(x)
        targets = target[None, None, :].expand(outputs.shape[:-1])

        loss = F.nll_loss(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for g in ensemble.g_plus:
                g.clamp_(g_min, g_max)
            for g in ensemble.g_minus:
                g.clamp_(g_min, g_max)

            count = targets.numel()
            total_loss += loss.item() * count
            total_correct += (
                outputs.argmax(dim=-1) == targets
            ).sum().item()
            total_predictions += count

        if batch_idx % log_interval == 0:
            print(
                f"Batch {batch_idx:4d}/{len(train_loader)}  "
                f"Loss: {total_loss / total_predictions:.4f}  "
                f"Accuracy: {100 * total_correct / total_predictions:.2f}%"
            )

    return (
        total_loss / total_predictions,
        total_correct / total_predictions,
    )


@torch.no_grad()
def evaluate_oi(ensemble, device, test_loader):
    ensemble.eval()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for x, target in test_loader:
        x, target = x.to(device), target.to(device)
        outputs = ensemble(x)
        targets = target[None, None, :].expand(outputs.shape[:-1])

        total_loss += F.nll_loss(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        ).item()
        total_correct += (
            outputs.argmax(-1) == targets
        ).sum().item()
        total_predictions += targets.numel()

    return {
        "loss": total_loss / total_predictions,
        "accuracy": total_correct / total_predictions,
    }


@torch.no_grad()
def make_stock_realization(
    digital_model, ensemble, rpu_config, t_inference, device
):
    """Evaluate optimized pairs through stock noise and inference code."""
    config = deepcopy(rpu_config)
    config.drift_compensation = GlobalDriftCompensation()

    # Our conductance inverse already returns full logical weights.
    config.mapping.weight_scaling_omega = 0.0
    config.mapping.learn_out_scaling = False

    stock_model = convert_to_analog(
        deepcopy(digital_model), rpu_config=config
    ).to(device)
    stock_model.eval()

    converter = SinglePairConductanceConverter(
        g_min=g_min, g_max=g_max
    )

    # Match our noise-model configuration, using the original methods.
    source_noise = ensemble.simulators[0].noise
    stock_noise = PCMLikeNoiseModel(
        g_converter=converter,
        prog_coeff=list(source_noise.prog_coeff),
        t_read=source_noise.t_read,
        t_0=source_noise.t_0,
        prog_noise_scale=source_noise.prog_noise_scale,
        read_noise_scale=source_noise.read_noise_scale,
        drift_scale=source_noise.drift_scale,
    )

    # Avoid the upstream constructor bug discussed earlier.
    stock_noise.prog_coeff_g_max_reference = (
        source_noise.prog_coeff_g_max_reference
    )

    scales = ensemble.simulators[0].scales

    for i, name in enumerate(("l1", "l2")):
        layer = getattr(stock_model, name)
        tiles = list(layer.analog_tiles())
        assert len(tiles) == 1
        tile = tiles[0]
        assert tile.digital_bias

        bias = getattr(digital_model, name).bias.detach()
        targets = [
            ensemble.g_plus[i].detach().to(device),
            ensemble.g_minus[i].detach().to(device),
        ]
        params = {"scale_ratio": scales[i]}

        programmed = [
            stock_noise.apply_programming_noise_to_conductance(g)
            for g in targets
        ]
        nus = [
            stock_noise.generate_drift_coefficients(g)
            for g in targets
        ]

        # Establish the reference after programming, before drift.
        reference_weights = converter.convert_back_to_weights(
            programmed, params
        )
        layer.set_weights(reference_weights, bias)

        compensation = tile.rpu_config.drift_compensation
        reference = compensation.init_baseline(tile)

        drifted = [
            stock_noise.apply_drift_noise_to_conductance(g, nu, t_inference)
            for g, nu in zip(programmed, nus)
        ]
        drifted_weights = converter.convert_back_to_weights(
            drifted, params
        )
        layer.set_weights(drifted_weights, bias)

        # Use the original calibration readout and correction.
        readout = tile._forward_drift_readout_tensor(
            False, exact_reference=False
        )
        tile.alpha = compensation.apply(readout, reference)

    return stock_model


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_nn.pt")

    rpu_config = InferenceRPUConfig(drift_compensation = None)
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0
    rpu_config.forward.is_perfect = False

    model.eval()

    compensation = oi.DifferentiableGlobalDriftCompensation()
    noise = oi.DifferentiablePCMLikeNoiseModel()

    n_realizations = 4
    t_inferences = [86400, 30 * 86400, 365 * 86400]  # 1 day, 30 days, 1 year
    device = next(model.parameters()).device

    weights = [model.l1.weight, model.l2.weight]
    scales = [
        oi.calculate_scale(w, g_max, g_min)
        for w in weights
    ]

    with torch.no_grad():
        pairs = [
            oi.logical_to_differential(w, s, g_min)
            for w, s in zip(weights, scales)
        ]

    simulators = [
        oi.PCMSimulator(
            convert_to_analog(model, rpu_config=rpu_config).to(device),
            weights,
            scales,
            noise,
            compensation,
        )
        for _ in range(n_realizations)
    ]

    ensemble = oi.PCMEnsemble(pairs, simulators, t_inferences)
    ensemble.eval()

    optimizer = torch.optim.Adam(
        list(ensemble.g_plus) + list(ensemble.g_minus),
        lr=0.01,
    )

    history = []

    log_interval = 10
    for epoch in range(1, 2):
        start = perf_counter()
        loss, accuracy = train_oi_epoch(ensemble, device, train_loader,
                                        optimizer, epoch, log_interval)
        end = perf_counter()
        history.append({"loss": loss, "accuracy": accuracy})

        print(
            f"Epoch {epoch}: loss={loss:.4f}, "
            f"accuracy={100 * accuracy:.2f}%, "
            f"time taken {end - start}, "
            f"evaluation={evaluate_oi(ensemble, device, test_loader)}"
        )

    for t in t_inferences:
        for realization in range(n_realizations):
            stock_model = make_stock_realization(
                model, ensemble, rpu_config, t, device
            )
            print(f"Stock evaluation: time={t}, realization={realization}")
            test(stock_model, device, test_loader)


if __name__ == '__main__':
    main()
