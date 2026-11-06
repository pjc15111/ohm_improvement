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

Digital training runs on GPUs for 14 epochs, followed by five OI epochs by default.
Each WPO job requests 64 physical CPU cores (`--hint=nomultithread`). Each CPU WPO task
can start as soon as its matching digital task succeeds. OI GPU tasks wait for
**all** digital tasks to succeed, so OI cannot hold GPUs needed by the remaining
baselines. No GPU type or concurrency limit is imposed. The scripts use the
repository's `venv` and current checkout.

Set `RUNS`, `DIGITAL_EPOCHS`, `OI_EPOCHS`, `HAT_EPOCHS` (default five),
`GPU_PARTITION` (default `general-gpu`),
`CPU_PARTITION` (default `general`), `GPU_TIME` (default `00:15:00`), or `WPO_TIME`
(default `04:00:00`) before submission to override settings. `MNIST_DATA` can
select an existing shared dataset directory; otherwise the scripts use `data`.
WPO requests 64 GB per job by default, independently of the GPU jobs' 16 GB;
override with `WPO_MEM`, for example `WPO_MEM=128G bash experiments/mnist/submit.sh`.
Spawned workers each load Python/AIHWKIT, so memory usage grows with worker count;
the larger default is a starting allocation, not a measured peak requirement.

The wrapper prints job IDs and a directory under `results/mnist/run-*`.
Each `seed-N` directory contains `digital.log`, `oi.log`, `wpo.log`, and their
history JSON files. Temporary checkpoints are removed by a final cleanup job,
even after a failed stage. Only logs and histories are retained.

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
Parallel workers are spawned with one compute thread each and independent
PyTorch/NumPy seed streams. Search/distribution sampling uses `--seed`;
PCM sampling/evaluation uses `--noise-seed`. Parallel scheduling and AIHWKIT's
separate backend RNG still prevent guaranteed bitwise repeatability.

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

Standalone HAT defaults to off (`--hat-epochs 0`); the SLURM wrapper runs five
HAT epochs by default. For example, `HAT_EPOCHS=0 bash experiments/mnist/submit.sh`
skips HAT. When enabled, it uses `AnalogSGD` with
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
