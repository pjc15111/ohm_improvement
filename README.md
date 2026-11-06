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
