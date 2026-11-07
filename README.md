# Ohm Improvement

## Model and reproducible PCM noise

Construct `OIModel` from an AIHWKIT model (`AnalogWrapper`, `AnalogSequential`,
or a standalone analog layer such as `AnalogLinear`):

```python
import ohm_improvement as oi

model = oi.OIModel(analog_model, seed=101)
model.program("train1")
model.drift(86400)
output = model(inputs)
stock_snapshot = model.to_analog()
```

`OIModel` copies the analog model by default; `inplace=True` converts it in place.
Its methods replace the former free functions in `model.py`.

Each tile owns one seed, derived from the model seed and tile path. Realization
IDs are arbitrary strings, and noise sides are `"plus"`/`"minus"`. `program(id)`
replaces one current realization; `drift(seconds)` always starts from its retained
programmed conductances and drift coefficients, without compounding drift.
Read noise is independently keyed by elapsed time. Explicit seeds and noise
regeneration do not consume the global PyTorch RNG; omitting the model seed
samples one from it. Direct `OITile` construction requires an explicit seed.

Reprogram after parameter updates or before starting a fresh backward graph.
For several backward calls sharing one programmed state, use `retain_graph=True`
until the last call, or accumulate losses and call backward once. MNIST follows
this rule. Stale state after a parameter update is rejected.

Seeds are included in `state_dict`. Restore into the same model structure and
conductance layout, then call `program()` and optionally `drift()` before inference.
This refactor changes checkpoint keys and noise streams; older checkpoints need
migration. Reproducibility assumes the same PyTorch backend, device, and dtype.

MNIST uses `--noise-seed` (default 101) with named training and test realizations,
replacing the three train/validation/evaluation seed options.

## Optimizer parameter groups

`OIModel.optimization_parameters()` returns named groups ready for Adam:

```python
optimizer = torch.optim.Adam(model.optimization_parameters(
    conductance_lr=0.01, significance_lr=0.0001,
    include_bias=False, include_output_scaling=False,
))
```

Defaults are `conductance_lr=0.01`, `significance_lr=0.0001`, `bias_lr=0.0001`,
and `output_scaling_lr=0.0001`. Inclusion switches are unchanged; frozen and
absent parameters are omitted. Override rates in this helper: an `lr` passed to
Adam does not replace explicit group rates. In MNIST, `--oi-lr` (also
`--conductance-lr`) sets only conductance LR; `--significance-lr` sets significance
LR. Bias and output scaling remain excluded from MNIST optimization.

## Convolution and mapped arrays

OI uses AIHWKIT's unfolded `Conv2d` path, including mapped convolutions and
linear layers split across input/output dimensions. Conversion disables indexing,
including when the source model has already run an indexed forward. Digital bias
owned by a mapped array is included when `include_bias=True`.

Compatibility tests cover residual CNN blocks, a two-layer AIHWKIT LSTM, and
BERT-style attention/projections with digital embeddings and layer norms. These
are small structural tests; full paper-model experiments remain to be added.

## Known Limitations

- Indexed execution is unsupported, including AIHWKIT's `Conv1d` and `Conv3d`.
  Unfolded `Conv2d` can require more memory than indexed simulation; AIHWKIT's
  restrictions on convolution groups and padding modes still apply.
- OI supports AIHWKIT `InferenceTile` with shared weights, stock `PCMLikeNoiseModel`,
  and global drift compensation (or no compensation). Other tile/noise types and
  compensation variants are unsupported; disabling the noise model is unsupported.
- Analog bias is unsupported: extraction/OI initialization drops its conductance
  column; the stock evaluator explicitly rejects it. Digital bias is supported.
- Conductance mappings must use differential pairs and scalar scaling; arbitrary
  converter representations, such as affine single-device mappings, are unsupported.
- Seeds control PCM noise, not AIHWKIT's backend forward/calibration RNG. Exact
  repeatability requires perfect forward I/O and the same backend/device/dtype.
- OI scale is fixed; trainable pair significance is clamped to `[0.1, 5]`, without
  hardware-discrete constraints. Conductance optimization can change nominal weights.
- The backend tile is held outside PyTorch submodule registration: move the analog
  model to its target device before OI conversion. Checkpoint restoration requires
  matching construction settings and reprogramming; old noise-buffer checkpoints
  need migration. AIHWKIT's initial digital-to-analog conversion may lose shared
  aliases; OI conversion preserves the aliases it receives.
- The evaluator retains one realization and a frozen scheme snapshot, mutates the
  supplied analog model, and requires matching tile paths and mapping scales.
- Experiments currently cover MNIST; paper-scale benchmarks/device models and WPO
  comparisons remain to be added. Installed stock WPO optimization runs on CPU;
  custom-converter initialization also uses CPU interpolation.

## Histories and results

All MNIST experiment code lives in `experiments/mnist`.
Train the shared digital baseline, then optimize OI:

```bash
python -m experiments.mnist.digital --epochs 3 --seed 1
python -m experiments.mnist.oi --seed 1
```

The standalone programs save to `results/mnist`: digital weights in `digital.pt`,
training history in `digital.json`, and OI history/evaluations in `oi.json`.
Use `--checkpoint` and `--results-file` to override these paths; `--help` lists
training and optimization options. Commit hash, device, metrics, and runtimes
are printed to the log.

To submit five paired seeds on SLURM:

```bash
bash experiments/mnist/submit.sh
```

The wrapper uses five seeds, 14 digital epochs, and five HAT epochs. It runs
six OI conditions per seed, with either digital or shared HAT initialization:

- 12 programming/time samples per optimizer update for 5 epochs: four independent
  realizations at each of the three training times.
- One programming/time sample per update for 60 epochs: choose a training time
  uniformly at random on each update.
- The same single-sample, 60-epoch run with learning rates reduced tenfold:
  conductance LR `0.001`, significance LR `0.00001`.

All three conditions draw fresh noise by default and have the same total number of
sample evaluations. Each sample follows program → drift → infer → backward;
losses are averaged before one optimizer update. Programming realizations are
independent across times as well as samples and updates. Each submission draws a fresh digital RNG seed base;
replicate N uses `seed_base + N` for initialization and training-data order.
Actual seeds appear in logs and history configuration. Directories `seed-N`
identify replicate indices, not the sampled seed values. `SEED_BASE` can be set
explicitly to reproduce a submission. Digital training remains noise-free, and
the network architecture is unchanged for this ablation.

HAT runs once per seed on GPU and saves a shared checkpoint. All three HAT+OI
conditions and CPU WPO load those exact weights. Conversion to analog tiles is
performed on CPU before transferring OI to GPU, preserving identical initial
tile weights, digital biases, and scales within each replicate. OI noise seeds
remain paired across conditions (`100 + replicate index`). All digital jobs finish before
HAT starts; all HAT jobs finish before any OI job starts. Each CPU WPO task can
start as soon as its own HAT checkpoint is ready. No GPU type or concurrency
limit is imposed. The scripts use the repository's `venv` and current checkout.

Set `RUNS`, `DIGITAL_EPOCHS`, `OI_EPOCHS` (default five for 12 samples),
`OI_SINGLE_EPOCHS` (default 60 for one sample), `HAT_EPOCHS` (default five),
`SEED_BASE` (random by default), `NOISE_MODE` (`fresh` by default, or `fixed`),
`GPU_PARTITION` (default `general-gpu`),
`CPU_PARTITION` (default `general`), `GPU_TIME`, or `WPO_TIME` before submission.
Both compute time limits default to **02:00:00**. `MNIST_DATA` can select an
existing shared dataset directory; otherwise the scripts use `data`.
WPO requests 64 physical CPU cores and 64 GB per job, independently of the GPU
jobs' 16 GB; override process/CPU count with `WPO_WORKERS` and memory with `WPO_MEM`. Spawned workers each load
Python/AIHWKIT, so memory usage grows with worker count.

The wrapper prints job IDs and a directory under `results/mnist/run-*`.
Each `seed-N` directory contains these log/history pairs:

- `digital.log` / `digital.json`
- `hat.log` / `hat.json`
- `wpo.log` / `wpo.json`
- `oi-digital-s12.log` / `oi-digital-s12.json`
- `oi-digital-s1.log` / `oi-digital-s1.json`
- `oi-digital-s1-lr0.1.log` / `oi-digital-s1-lr0.1.json`
- `oi-hat-s12.log` / `oi-hat-s12.json`
- `oi-hat-s1.log` / `oi-hat-s1.json`
- `oi-hat-s1-lr0.1.log` / `oi-hat-s1-lr0.1.json`

Digital and HAT checkpoints are temporary handoffs. A final cleanup job removes
them after every stage terminates, leaving only logs and histories.

OI's `results` retain the ordinary stock-noise comparisons on the test dataset.
The separate `diagnostics` field records the Cartesian product of
`data_split = seen/unseen` and `realization_split = seen/unseen`, at each inference
time. Seen data is the full MNIST training dataset; unseen data is the test
split. All conditions evaluate four realizations per group. At training times,
seen IDs are selected evenly from IDs actually trained at that specific time,
so programming, drift, and read samples were all seen. At held-out times,
previously trained programming realizations receive fresh time-dependent read
noise. Unseen realizations use a separate `testN` namespace.

Training times are 1 day, 30 days, and 1 year. Final evaluations additionally
cover 7 days and 90 days (interpolation), and 3 years (extrapolation). Both ordinary
results and diagnostics include `time_split = seen/unseen`. Held-out times do not
enter OI training or the WPO objective/progress evaluations; they are evaluated
only after optimization. Override these lists with `--times` and `--unseen-times`
in the standalone OI/WPO programs. Passing `--unseen-times` with no values
omits the additional times.

Diagnostics generate exact OI realization states and export stock analog
snapshots (`simulator = stock_snapshot`) for inference on both data splits.
This is necessary because stock evaluation and OI use different seed derivations:
passing the name `train1` to the stock evaluator would not reproduce OI's
training noise. Backend forward/calibration randomness remains uncontrolled.

For standalone ablations, use `--noise-mode fresh` (default) or `fixed`,
`--train-realizations`, `--time-sampling all/random`, and
`--evaluation-realizations`. In all-time mode, `--train-realizations` is the number
of independent samples at each time; in random-time mode it is the number per
optimizer update. Fresh IDs include the time index and global optimizer step,
making them reproducible without retaining noise tensors. Time sampling has its
own seeded RNG and continues across epoch boundaries. To
prepare shared HAT weights directly:

```bash
python -m experiments.mnist.hat --hat-epochs 5
python -m experiments.mnist.oi --hat-checkpoint results/mnist/hat.pt --train-realizations 16
```

The WPO command loads the same checkpoint, optionally fine-tunes a copy with HAT,
then runs stock AIHWKIT WPO on CPU. It learns one shared conductance converter
from the pooled, per-tile-normalized nominal weight distribution. Bias and output
scaling are excluded from that distribution. Final stock evaluations compare
`baseline` and `wpo`, plus `hat` when HAT is enabled, at the same `testN`
realizations/times and noise seed as OI. The original models are preserved.

```bash
python -m experiments.mnist.wpo --no-accel --hat-epochs 5
```

WPO defaults to 200 generations, population multiplier 100, six positive-side
bins, and 100 baseline-loss samples, retaining stock stopping/polishing settings.
Its synthetic test array is 512 inputs by 20 outputs; this is independent of the
inference model's unmapped-array setting. Use `--wpo-maxiter`, `--wpo-popsize`,
`--wpo-bins`, `--wpo-baseline-iterations`, `--wpo-input-size`, `--wpo-output-size`,
and `--no-wpo-polish` to change search cost. Pair factors are fixed by default;
`--optimize-significance` lets WPO optimize shared factors. `--dry-run` uses a
small one-generation search with polishing disabled.

`--workers` defaults to `SLURM_CPUS_PER_TASK` (64 in the wrapper), otherwise one.
Parallel workers are spawned with one compute thread each. Each objective
evaluation receives an independent seed assigned in the parent process, without
a shared-memory seed counter. This changes the noise streams from earlier runs.
Search/distribution sampling uses `--seed`;
PCM sampling/evaluation uses `--noise-seed`. Parallel scheduling and AIHWKIT's
separate backend RNG still prevent guaranteed bitwise repeatability.

Python's process pool still needs operating-system semaphore resources. If it
fails with `No space left on device`, check `/dev/shm` free space and inodes on
the allocated compute node. Raising the job's RAM request does not by itself
increase that filesystem's capacity. `WPO_WORKERS=1 WPO_MEM=8G bash experiments/mnist/submit.sh`
requests a single CPU worker and bypasses multiprocessing as a temporary
workaround; parallel operation may require HPC support to resolve node resource
exhaustion.

The callback records stock objective/TNMSE each generation. It evaluates the
first generation and then every `--evaluation-interval` optimization seconds
(default 60), checked at generation boundaries. `--progress-realizations`
defaults to one; progress uses `validationN` noise realizations, separate from
final `testN` realizations. Classification metrics are reporting only, without
changing WPO's weight-error objective or stopping rule. Progress-evaluation time
is tracked separately and excluded from history timestamps for optimization.

History is saved after every generation so interrupted jobs retain progress.
`config.stage` records `optimization`, `evaluation`, or `complete`. The final
converter's pair factors and conductance curves are included in
`config.wpo_converter`; no additional result artifacts are written.

In-process HAT in the WPO program defaults to off (`--hat-epochs 0`); the SLURM wrapper runs five
HAT epochs by default. For example, `HAT_EPOCHS=0 bash experiments/mnist/submit.sh`
skips HAT updates. `--hat-checkpoint` loads shared pretrained weights in OI or
WPO. HAT uses `AnalogSGD` with
learning rate 0.01, momentum 0.9, and epoch LR decay 0.7. Training weight noise
has standard deviation 0.1 relative to each tile's maximum, with clipping at
2.5 standard deviations. These are configurable starting settings, not a
paper-matched recipe. See `--hat-lr`, `--hat-momentum`, `--hat-gamma`,
`--hat-weight-noise`, and `--hat-clip-sigma`. Test-set metrics during HAT are
reporting only, without early stopping or model selection. The original baseline
is preserved; the prepared model returns to CPU with HAT modifiers disabled.

History records hold named `metrics` and optional batch/epoch/sample information,
so optimization steps can also represent WPO generations. Each experiment saves
its `method` and `optimization_history`; digital training history is optional for
jobs loading a shared checkpoint. This replaces the old `oi_history` JSON key and
moves record loss/accuracy into `metrics`.
