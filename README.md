# Ohm Improvement

## Reproducible PCM noise

Pass independent seeds when converting an AIHWKIT model:

```python
model = oi.convert_to_oi(
    analog_model, n_realizations=4, t_inferences=[0, 86400],
    noise_seeds={"train": 101, "validation": 202, "evaluation": 303},
)
oi.program_weights(model, realization_index=0, time_index=1,
                   noise_split="validation")
```

`noise_split` defaults to `"train"`; it is independent of `model.train()` and
`model.eval()`. All three streams support indices `0 .. n_realizations-1`.
Seeds must be distinct integers in `[0, 2**63)`. Omitting them samples seeds at
conversion using PyTorch's global RNG. Direct `OITile` construction requires
explicit, valid seeds; seed initialization and validation live in `convert_to_oi`. Explicit seeds and subsequent sample
regeneration do not consume the global PyTorch RNG.

Each tile stores only three seeds. Module paths distinguish tiles, and samples
are regenerated on the conductance device and dtype. Programming and drift
randomness is shared across times in a realization; read randomness is keyed by
elapsed time. Calls can be repeated or reordered without changing the samples.
The noisy conductances are recomputed from current parameters with autograd.

Seeds are included in `state_dict`. Restore into a model with the same structure,
conductance layout, realization count, and inference times, then call
`program_weights` before inference. Reproducibility assumes the same PyTorch
backend, device type, and dtype; CPU and CUDA samples need not agree. Older
checkpoints containing full noise buffers require migration before strict loading.

`experiments/mnist.py` exposes `--train-noise-seed`, `--validation-noise-seed`,
and `--evaluation-noise-seed` (defaults 101, 202, 303).

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
- OITile still uses integer realization/time indices and three named noise streams.
  It recomputes programming on each call; separate `program()`/`drift()` operations,
  string realization IDs/sides, and one base seed remain planned.
- OI scale is fixed; trainable pair significance is clamped to `[0.1, 5]`, without
  hardware-discrete constraints. Conductance optimization can change nominal weights.
- The backend tile is held outside PyTorch submodule registration: move the analog
  model to its target device before OI conversion. Checkpoint restoration requires
  matching construction settings and reprogramming; old noise-buffer checkpoints
  need migration. Shared-module aliases are not preserved by OI conversion.
- The evaluator retains one realization and a frozen scheme snapshot, mutates the
  supplied analog model, and requires matching tile paths and mapping scales.
- Experiments currently cover MNIST; paper-scale benchmarks/device models and WPO
  comparisons remain to be added. Installed stock WPO optimization runs on CPU;
  custom-converter initialization also uses CPU interpolation.

## Histories and plots

MNIST saves `history.json` and PNG/PDF figures under a timestamped
`results/mnist/` directory; set `--output-dir PATH` to choose the location.
Histories contain run configuration, digital batch/epoch metrics, OI metrics for
every realization/time at each optimizer step, and final stock evaluations.

Plots include digital loss/accuracy; optimization loss/accuracy overall, by drift
time, by realization, and by both; and final stock inference curves by method.
Optimization plots show minibatch training metrics before each update, not
held-out performance. Dotted lines show the final digital evaluation baseline;
loss figures include digital loss and digital accuracy on a labeled right axis.
Future WPO evaluation records will appear alongside other methods automatically.

Use `--plot-x seconds|step|epoch` to select the optimization x-axis (default:
seconds). Regenerate plots without retraining:

```bash
python -m experiments.plotting results/mnist/RUN/history.json --x-axis step
```
