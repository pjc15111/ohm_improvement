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


def train_oi_epoch(oi_model, realizations, t_inferences, device,
                   train_loader, optimizer, log_interval):
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)

        n_samples = len(realizations) * len(t_inferences)
        count = target.numel()

        for realization in realizations:
            for t_i in range(len(t_inferences)):
                oi.program_weights(oi_model, realization, t_i)
                output = oi_model(x)
                sample_loss = F.nll_loss(output, target)

                (sample_loss / n_samples).backward()

                total_loss += sample_loss.item() * count
                total_correct += (
                    output.argmax(dim=-1) == target
                ).sum().item()
                total_predictions += count

        optimizer.step()
        oi.clamp_model(oi_model)

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
def evaluate_oi(oi_model, realizations,
                t_inferences, device, test_loader, noise_split="evaluation"):
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for realization in realizations:
        for t_i in range(len(t_inferences)):
            oi.program_weights(oi_model, realization, t_i, noise_split=noise_split)
            for x, target in test_loader:
                x, target = x.to(device), target.to(device)
                output = oi_model(x)

                total_loss += F.nll_loss(
                    output,
                    target,
                    reduction="sum",
                ).item()
                total_correct += (
                    output.argmax(dim=-1) == target
                ).sum().item()
                total_predictions += target.numel()

    return {
        "loss": total_loss / total_predictions,
        "accuracy": total_correct / total_predictions,
    }


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
    parser.add_argument('--train-noise-seed', type=int, default=101)
    parser.add_argument('--validation-noise-seed', type=int, default=202)
    parser.add_argument('--evaluation-noise-seed', type=int, default=303)
    args = parser.parse_args()
    noise_seeds = dict(train=args.train_noise_seed,
                       validation=args.validation_noise_seed,
                       evaluation=args.evaluation_noise_seed)
    if len(set(noise_seeds.values())) != 3 or any(
            not 0 <= seed < 2**63 for seed in noise_seeds.values()):
        parser.error('noise seeds must be distinct integers in [0, 2**63)')

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

    rpu_config = InferenceRPUConfig()
    rpu_config.drift_compensation = GlobalDriftCompensation()
    rpu_config.mapping.max_input_size = 0
    rpu_config.mapping.max_output_size = 0
    rpu_config.forward.is_perfect = False

    model.eval()

    n_realizations = 4
    n_evaluation_realizations = 4
    t_inferences = [86400, 30 * 86400, 365 * 86400]  # 1 day, 30 days, 1 year
    device = next(model.parameters()).device

    analog_model = convert_to_analog(
        model, rpu_config=rpu_config
    ).to(device)

    oi_model = oi.convert_to_oi(
        analog_model,
        n_realizations=n_realizations + n_evaluation_realizations,
        t_inferences=t_inferences,
        noise_seeds=noise_seeds,
    )
    oi_model.eval()

    conductances = oi.optimization_parameters(
        oi_model,
        include_conductances=True,
        include_output_scaling=False,
        include_bias=False
    )

    optimizer = torch.optim.Adam(conductances, lr=0.01)

    history = []

    log_interval = 10
    for epoch in range(1, 2):
        start = perf_counter()
        loss, accuracy = train_oi_epoch(oi_model, range(n_realizations), t_inferences,
                                        device, train_loader,
                                        optimizer, log_interval)
        end = perf_counter()
        history.append({"loss": loss, "accuracy": accuracy})

        print(
            f"Epoch {epoch}: loss={loss:.4f}, "
            f"accuracy={100 * accuracy:.2f}%, "
            f"time taken {end - start}, "
            f"evaluation={evaluate_oi(
                              oi_model,
                              range(n_realizations, n_realizations + n_evaluation_realizations),
                              t_inferences,
                              device, test_loader
                          )}"
        )

    for realization in range(n_realizations + n_evaluation_realizations):
        for t_i in range(len(t_inferences)):
            noise_split = "train" if realization < n_realizations else "evaluation"
            oi.program_weights(oi_model, realization, t_i, noise_split=noise_split)
            stock_model = oi.oi_to_analog(oi_model)
            print(f"Stock evaluation: time={t_inferences[t_i]}, realization={realization}")
            test(stock_model, device, test_loader)


if __name__ == '__main__':
    main()
