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

## Known Limitations

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

MNIST retains dataclass history records for digital training and every OI
realization/time at each optimizer step. Final stock evaluations and histories
are saved to `results/mnist.json`; use `--results-file PATH` to choose a file.
The commit hash, device, metrics, and runtimes are printed to the log.

For 15 independent SLURM runs on available GPUs:

```bash
bash scripts/submit_mnist_gpu_comparison.sh
```

Each run writes `results/mnist/JOB_TASK.log` and `JOB_TASK.json`. The runner uses
the current checkout. There are no source archives, manifests, or automatic plots.
