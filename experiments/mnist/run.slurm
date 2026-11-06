#!/usr/bin/env bash
# Resource requests and dependencies are supplied by submit.sh.
set -euo pipefail

phase=$1
run_dir=$2
seed=${SLURM_ARRAY_TASK_ID:-0}
printf 'Commit: %s\nPhase: %s\nSeed: %s\n' "$(git rev-parse HEAD)" "$phase" "$seed"

if [[ $phase == cleanup ]]; then
    rm -f -- "$run_dir"/checkpoints/*.pt
    rmdir -- "$run_dir/checkpoints"
    exit 0
fi

if type module >/dev/null 2>&1; then
    module load python/3.12.14
fi
python_bin=$PWD/venv/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

checkpoint=$run_dir/checkpoints/seed-$seed.pt
results=$run_dir/seed-$seed/$phase.json
data_dir=${MNIST_DATA:-$PWD/data}
args=(--seed "$seed" --checkpoint "$checkpoint" --results-file "$results" --data-dir "$data_dir")

case "$phase" in
    digital)
        "$python_bin" -m experiments.mnist.digital "${args[@]}" --epochs "$3"
        ;;
    oi)
        "$python_bin" -m experiments.mnist.oi "${args[@]}" --noise-seed "$((100 + seed))" --oi-epochs "$3"
        ;;
    wpo)
        "$python_bin" -m experiments.mnist.wpo "${args[@]}" --noise-seed "$((100 + seed))" --no-accel --hat-epochs "$3"
        ;;
    *)
        echo "Unknown MNIST phase: $phase" >&2
        exit 1
        ;;
esac
