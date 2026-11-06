"""Optimize MNIST programming with stock CPU WPO, optionally fine-tuning with HAT.

Load the same digital checkpoint used by OI, optimize its programming mapping,
and evaluate through experiments.common's stock evaluator with matching seeds,
realizations, and inference times. Shared MNIST setup lives in experiments.mnist.common;
experiments.mnist.digital trains the baseline.
"""

import argparse
from argparse import Namespace
from copy import deepcopy
from contextlib import contextmanager, nullcontext
from functools import partial
import math
import multiprocessing
import os
import random
import warnings
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from aihwkit.inference.converter.conductance import CustomPairConductanceConverter
from aihwkit.inference.converter.wpo import WeightProgrammingOptimizer, denormalize, stop_criterion
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.enums import WeightClipType, WeightModifierType

from experiments.common import evaluate_classifier, evaluate_programming, evaluator_from_analog
from experiments.history import ExperimentHistory, HistoryRecord
from experiments.mnist.common import (
    load_digital_model, log_commit, make_loaders, make_rpu_config, report_metrics,
    select_device, synchronize_device,
)
from ohm_improvement import AnalogModel


def parse_args(argv=None) -> Namespace:
    """Configure checkpoint, data, WPO, evaluation, and results/mnist/wpo.json."""
    parser = argparse.ArgumentParser(description='MNIST stock CPU WPO optimization and evaluation')
    parser.add_argument('--checkpoint', default='results/mnist/digital.pt')
    parser.add_argument('--results-file', default='results/mnist/wpo.json')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--noise-seed', type=int, default=101)
    parser.add_argument('--no-accel', action='store_true', help='run optional HAT on CPU too')
    parser.add_argument('--dry-run', action='store_true', help='one batch and at most one HAT epoch')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--pair-factors', type=float, nargs='+', default=[1.])
    parser.add_argument('--times', type=float, nargs='+', default=[86400., 30 * 86400., 365 * 86400.])
    parser.add_argument('--evaluation-realizations', type=int, default=4)
    parser.add_argument('--hat-epochs', type=int, default=0, help='optional HAT epochs; zero skips HAT')
    parser.add_argument('--hat-lr', type=float, default=.01)
    parser.add_argument('--hat-momentum', type=float, default=.9)
    parser.add_argument('--hat-gamma', type=float, default=.7)
    parser.add_argument('--hat-weight-noise', type=float, default=.1,
                        help='Gaussian weight-noise std relative to each tile maximum')
    parser.add_argument('--hat-clip-sigma', type=float, default=2.5)
    parser.add_argument('--workers', type=int, default=None,
                        help='WPO processes (default: SLURM_CPUS_PER_TASK, otherwise one)')
    parser.add_argument('--wpo-maxiter', type=int, default=200)
    parser.add_argument('--wpo-popsize', type=int, default=100)
    parser.add_argument('--wpo-baseline-iterations', type=int, default=100)
    parser.add_argument('--wpo-bins', type=int, default=6)
    parser.add_argument('--wpo-input-size', type=int, default=512)
    parser.add_argument('--wpo-output-size', type=int, default=20)
    parser.add_argument('--wpo-loss-margin', type=float, default=.125)
    parser.add_argument('--wpo-polish', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wpo-symmetric', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--optimize-significance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--evaluation-interval', type=float, default=60.,
                        help='optimization seconds between progress evaluations; zero evaluates every generation')
    parser.add_argument('--progress-realizations', type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers is None:
        args.workers = 1 if args.dry_run else int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    if any(value <= 0 for value in (args.batch_size, args.test_batch_size,
                                   args.log_interval, args.evaluation_realizations, args.progress_realizations,
                                   args.workers, args.wpo_maxiter, args.wpo_popsize,
                                   args.wpo_baseline_iterations, args.wpo_input_size, args.wpo_output_size)):
        parser.error('batch sizes, intervals/counts, workers, and WPO search sizes must be positive')
    if args.wpo_bins < 2:
        parser.error('WPO needs at least two positive-side bins')
    if not math.isfinite(args.wpo_loss_margin) or not 0 <= args.wpo_loss_margin < 1:
        parser.error('WPO loss margin must be finite and in [0, 1)')
    if args.hat_epochs < 0:
        parser.error('HAT epochs must be nonnegative')
    if any(not 0 <= seed < 2**63 for seed in (args.seed, args.noise_seed)):
        parser.error('seeds must be integers in [0, 2**63)')
    if any(not math.isfinite(value) or value <= 0 for value in
           [*args.pair_factors, args.hat_lr, args.hat_gamma, args.hat_clip_sigma]):
        parser.error('pair factors, HAT learning rate, gamma, and clip sigma must be finite and positive')
    if any(not math.isfinite(value) or value < 0 for value in
           [*args.times, args.hat_weight_noise, args.hat_momentum, args.evaluation_interval]):
        parser.error('times, HAT noise/momentum, and evaluation interval must be finite and nonnegative')
    if args.dry_run:
        args.hat_epochs = min(args.hat_epochs, 1)
        args.evaluation_realizations = 1
        args.times = args.times[:1]
        args.progress_realizations = 1
        args.wpo_maxiter = 1
        args.wpo_popsize = 2
        args.wpo_baseline_iterations = 2
        args.wpo_bins = 3
        args.wpo_input_size, args.wpo_output_size = 16, 4
        args.wpo_polish = False
        args.evaluation_interval = 0.
    return args


def load_baseline(args: Namespace, test_loader) -> tuple[AnalogModel, dict]:
    """Load the shared digital checkpoint and construct the CPU analog baseline."""
    model = load_digital_model(args.checkpoint, torch.device('cpu'))
    metrics = evaluate_classifier(model, test_loader, torch.device('cpu'))
    report_metrics('Digital baseline', metrics)
    analog = convert_to_analog(model, rpu_config=make_rpu_config(args.pair_factors))
    return analog.eval(), metrics


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
            report_metrics(f'HAT batch {index}/{len(loader)}', batches[-1].metrics)
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


@contextmanager
def preserve_rng():
    """Isolate CPU PyTorch, NumPy, and Python RNGs; backend RNG is separate."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


@contextmanager
def single_threaded_wpo():
    """Use process parallelism without multiplying it by BLAS/OpenMP threads."""
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    threads = torch.get_num_threads()
    try:
        os.environ.update(dict.fromkeys(names, '1'))
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(threads)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def initialize_worker(noise_seed, counter):
    torch.set_num_threads(1)
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    seed = int(np.random.SeedSequence([noise_seed, index]).generate_state(1, dtype=np.uint64)[0]) % 2**63
    torch.random.default_generator.manual_seed(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


class WPOWorkerMap:
    """Spawned workers for SciPy, without pickling the pool into WPO's losses."""

    def __init__(self, pool):
        self.pool = pool

    def __call__(self, function, candidates):
        return self.pool.map(function, candidates)

    def __getstate__(self):
        return {}


def candidate_converter(optimizer, candidate) -> CustomPairConductanceConverter:
    """Decode the current candidate exactly as stock WPO decodes its result."""
    factors, conductances = denormalize(candidate.copy(), optimizer.__dict__.copy())
    return CustomPairConductanceConverter(
        f_lst=factors, g_lst=conductances, g_min=optimizer.g_min, g_max=optimizer.g_max,
        invertibility_test=False,
    )


class WPOHistoryCallback:
    """Record generation objectives and periodically evaluate the best converter.

    Decode candidates using WPO's denormalize(), preserve evaluation RNG state,
    account for evaluation time separately, and retain WPO's stopping criterion.
    """

    def __init__(self, optimizer, evaluate, evaluation_interval: float, *, on_progress=None, started=None):
        """Retain the optimizer and evaluation callback in the parent process."""
        self.optimizer = optimizer
        self.evaluate = evaluate
        self.evaluation_interval = evaluation_interval
        self.on_progress = on_progress
        self.started = perf_counter() if started is None else started
        self.evaluation_seconds = 0.
        self.last_evaluation = -math.inf
        self.step = 0
        self.history = []

    def seconds(self):
        return perf_counter() - self.started - self.evaluation_seconds

    def __call__(self, intermediate_result) -> bool:
        """Record one generation; return whether optimization should stop."""
        self.step += 1
        seconds = self.seconds()
        objective = float(intermediate_result.fun)
        baseline = float(self.optimizer.loss_baseline)
        tnmse = (objective - self.optimizer.loss_margin) * baseline + baseline
        self.history.append(HistoryRecord(
            metrics=dict(objective=objective, tnmse=tnmse), step=self.step, seconds=seconds,
        ))
        try:
            if self.evaluate is not None and seconds - self.last_evaluation >= self.evaluation_interval:
                self.evaluate_candidate(candidate_converter(self.optimizer, intermediate_result.x), seconds)
        finally:
            if self.on_progress is not None:
                self.on_progress(self.history)
        stop_criterion(intermediate_result)  # Preserve stock WPO's target-based termination.
        return False

    def evaluate_candidate(self, converter, seconds):
        started = perf_counter()
        try:
            with preserve_rng():
                records = self.evaluate(converter)
            for record in records:
                self.history.append(HistoryRecord(
                    metrics={key: float(record[key]) for key in ('loss', 'accuracy')},
                    count=record['count'], step=self.step, seconds=seconds,
                    realization_id=record['realization_id'], t_inference=record['t_inference'],
                ))
                report_metrics(
                    f"WPO generation {self.step} ({seconds:.2f}s) {record['realization_id']} "
                    f"t={record['t_inference']}s", record,
                )
        finally:
            self.evaluation_seconds += perf_counter() - started
            self.last_evaluation = seconds

    def __getstate__(self) -> dict:
        """Exclude live optimizer/evaluation machinery from WPO's pickled state."""
        return {}


def collect_wpo_weights(analog_model: AnalogModel) -> torch.Tensor:
    """Pool nominal weights, normalized per tile to match conductance conversion."""
    weights = []
    for tile in analog_model.analog_tiles():
        if tile.is_cuda:
            raise ValueError('Stock WPO requires a CPU analog model')
        raw, _ = tile.get_weights(apply_weight_scaling=False)
        raw = raw.detach().flatten()
        weights.append(raw / raw.abs().max().clamp_min(1e-7))
    if not weights:
        raise ValueError('WPO requires analog tiles')
    weights = torch.cat(weights)
    if not torch.isfinite(weights).all() or weights.abs().max() == 0:
        raise ValueError('WPO requires finite, nonzero weights')
    return weights


def optimize_wpo(
    args: Namespace, analog_model: AnalogModel, evaluate_progress, *, on_progress=None,
) -> tuple[CustomPairConductanceConverter, list[HistoryRecord]]:
    """Run stock WPO with generation history and optional model evaluations."""
    started = perf_counter()
    weights = collect_wpo_weights(analog_model)
    config = deepcopy(next(analog_model.analog_tiles()).rpu_config)
    print(f'WPO: workers={args.workers}, test array={args.wpo_input_size}x{args.wpo_output_size}, '
          f'maxiter={args.wpo_maxiter}, popsize={args.wpo_popsize}', flush=True)
    with preserve_rng(), single_threaded_wpo():
        np.random.seed(args.seed % 2**32)
        torch.random.default_generator.manual_seed(args.noise_seed)
        context = multiprocessing.get_context('spawn')
        pool_context = (context.Pool(
            args.workers, initializer=initialize_worker,
            initargs=(args.noise_seed, context.Value('i', 0)),
        ) if args.workers > 1 else nullcontext(None))
        with pool_context as pool:
            optimizer = WeightProgrammingOptimizer(
                weights, [None] * len(args.pair_factors) if args.optimize_significance else args.pair_factors,
                config, args.times, deepcopy(config.noise_model.g_converter),
                symmetric=args.wpo_symmetric,
                test_in_features=args.wpo_input_size, test_out_features=args.wpo_output_size,
                nbins=args.wpo_bins, maxiter=args.wpo_maxiter, popsize=args.wpo_popsize,
                baseline_iterations=args.wpo_baseline_iterations, loss_margin=args.wpo_loss_margin,
                polish=args.wpo_polish, seed=args.seed % 2**32,
                workers=WPOWorkerMap(pool) if pool is not None else 1,
            )
            callback = WPOHistoryCallback(
                optimizer, evaluate_progress, args.evaluation_interval,
                on_progress=on_progress, started=started,
            )
            optimizer.callback = callback
            # Stock WPO requests immediate updates for custom worker maps.
            # SciPy correctly enforces deferred updates; silence only its notice.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=UserWarning,
                    message="^differential_evolution: the 'workers' keyword has overridden "
                            "updating='immediate' to updating='deferred'$",
                )
                converter, _ = optimizer.run_optimizer()
        seconds = callback.seconds()
        callback.history.append(HistoryRecord(
            metrics=dict(tnmse=float(optimizer.best_loss), baseline_tnmse=float(optimizer.loss_baseline),
                         evaluation_seconds=callback.evaluation_seconds,
                         wall_seconds=perf_counter() - started),
            step=callback.step, seconds=seconds,
        ))
        print(f'WPO: baseline TNMSE={optimizer.loss_baseline:.6g}, final TNMSE={optimizer.best_loss:.6g}, '
              f'optimization={seconds:.2f}s, progress evaluation={callback.evaluation_seconds:.2f}s', flush=True)
        if on_progress is not None:
            on_progress(callback.history)
    return converter, callback.history


def evaluate_candidate(args, analog_model, loader, converter) -> list[dict]:
    """Use independent named PCM realizations for progress reporting only."""
    evaluator = evaluator_from_analog(deepcopy(analog_model), converter=converter, seed=args.noise_seed)
    return evaluate_programming(
        evaluator, partial(evaluate_classifier, loader=loader, device=torch.device('cpu')),
        [f'validation{i + 1}' for i in range(args.progress_realizations)], args.times,
    )


def run_comparison(
    args: Namespace, analog_model: AnalogModel,
    converter: CustomPairConductanceConverter, loader, *, baseline_model: AnalogModel | None = None,
) -> list[dict]:
    """Evaluate baseline/WPO mappings through the shared stock evaluator."""
    evaluators = {'baseline': evaluator_from_analog(
        deepcopy(analog_model if baseline_model is None else baseline_model), seed=args.noise_seed,
    )}
    if args.hat_epochs:
        evaluators['hat'] = evaluator_from_analog(deepcopy(analog_model), seed=args.noise_seed)
    evaluators['wpo'] = evaluator_from_analog(
        deepcopy(analog_model), converter=converter, seed=args.noise_seed,
    )
    evaluate = partial(evaluate_classifier, loader=loader, device=torch.device('cpu'))
    realization_ids = [f'test{i + 1}' for i in range(args.evaluation_realizations)]
    results = []
    for method, evaluator in evaluators.items():
        for record in evaluate_programming(evaluator, evaluate, realization_ids, args.times):
            results.append(dict(method=method, **record))
            report_metrics(f"{method} {record['realization_id']} t={record['t_inference']}s", record)
    return results


def main(argv=None) -> ExperimentHistory:
    """Load baseline, optionally run HAT, optimize WPO, evaluate, and save history."""
    args = parse_args(argv)
    log_commit()
    start = perf_counter()
    torch.manual_seed(args.seed)
    device = select_device(args.no_accel)
    print(f'HAT device: {device}; WPO device: cpu', flush=True)
    train_loader, test_loader = make_loaders(args, device)
    baseline, metrics = load_baseline(args, test_loader)
    # The test loader supplies reporting only; no early stopping/model selection.
    prepared, hat_history = train_hardware_aware(args, baseline, train_loader, test_loader, device)
    history = ExperimentHistory(
        method='wpo', config={**vars(args), 'stage': 'optimization'},
        digital_baseline=metrics, optimization_history=hat_history,
    )
    history.save(args.results_file)

    def save_progress(records):
        history.optimization_history = hat_history + records
        history.save(args.results_file)

    converter, records = optimize_wpo(
        args, prepared, partial(evaluate_candidate, args, prepared, test_loader),
        on_progress=save_progress,
    )
    history.optimization_history = hat_history + records
    history.config.update(stage='evaluation', wpo_converter=dict(
        f_lst=[float(f) for f in converter.f_lst],
        g_lst=[[float(g) for g in row] for row in converter.g_lst],
        g_min=float(converter.g_min), g_max=float(converter.g_max),
    ))
    history.save(args.results_file)
    history.results = run_comparison(args, prepared, converter, test_loader, baseline_model=baseline)
    history.config['stage'] = 'complete'
    history.save(args.results_file)
    print(f'Results: {args.results_file}')
    print(f'Total runtime: {perf_counter() - start:.2f}s')
    return history


if __name__ == '__main__':
    main()
