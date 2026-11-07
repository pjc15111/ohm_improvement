#!/usr/bin/env bash
# Resource requests and dependencies are supplied by submit.sh.
set -euo pipefail

phase=$1
run_dir=$2
seed_base=$3
shift 3
replicate=${SLURM_ARRAY_TASK_ID:-0}
seed=$((seed_base + replicate))
printf 'Commit: %s\nPhase: %s\nReplicate: %s\nSeed base: %s\nSeed: %s\n' \
    "$(git rev-parse HEAD)" "$phase" "$replicate" "$seed_base" "$seed"

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

checkpoint=$run_dir/checkpoints/seed-$replicate.pt
label=$phase
if [[ $phase == oi ]]; then
    label="oi-$2-e$1"
elif [[ $phase == wpo ]]; then
    label="wpo-$1"
fi
results=$run_dir/seed-$replicate/$label.json
hat_checkpoint=$run_dir/checkpoints/hat-seed-$replicate.pt
data_dir=${MNIST_DATA:-$PWD/data}
args=(--seed "$seed" --checkpoint "$checkpoint" --results-file "$results" --data-dir "$data_dir")

case "$phase" in
    digital)
        "$python_bin" -m experiments.mnist.digital "${args[@]}" --epochs "$1"
        ;;
    hat)
        "$python_bin" -m experiments.mnist.hat "${args[@]}" --hat-epochs "$1" --output-checkpoint "$hat_checkpoint"
        ;;
    oi)
        if [[ $2 == hat ]]; then
            args+=(--hat-checkpoint "$hat_checkpoint")
        fi
        "$python_bin" -m experiments.mnist.oi "${args[@]}" --noise-seed "$((100 + replicate))" \
            --oi-epochs "$1" --train-realizations 1 --time-sampling random \
            --conductance-lr 0.001 --significance-lr 0.00001 \
            --bias-lr 0.00001 --output-scaling-lr 0.00001 \
            --optimize-bias --optimize-output-scaling \
            --evaluation-realizations 4 --noise-mode "$3"
        ;;
    wpo)
        if [[ $1 == hat ]]; then
            args+=(--hat-checkpoint "$hat_checkpoint")
        fi
        "$python_bin" -m experiments.mnist.wpo "${args[@]}" --noise-seed "$((100 + replicate))" \
            --no-accel --workers "${WPO_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
        ;;
    *)
        echo "Unknown MNIST phase: $phase" >&2
        exit 1
        ;;
esac
