#!/usr/bin/env bash
# Submit independent digital, WPO, and OI arrays; no GPU type/concurrency cap.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runs=${RUNS:-5}
epochs=${DIGITAL_EPOCHS:-14}
oi_epochs=${OI_EPOCHS:-5}
hat_epochs=${HAT_EPOCHS:-5}
gpu_partition=${GPU_PARTITION:-general-gpu}
cpu_partition=${CPU_PARTITION:-general}
gpu_time=${GPU_TIME:-00:15:00}
wpo_time=${WPO_TIME:-04:00:00}
wpo_mem=${WPO_MEM:-64G}

if [[ ! $runs =~ ^[1-9][0-9]*$ || ! $epochs =~ ^[1-9][0-9]*$ || ! $oi_epochs =~ ^[1-9][0-9]*$ ]]; then
    echo 'RUNS, DIGITAL_EPOCHS, and OI_EPOCHS must be positive integers' >&2
    exit 1
fi
if [[ ! $hat_epochs =~ ^[0-9]+$ ]]; then
    echo 'HAT_EPOCHS must be a nonnegative integer' >&2
    exit 1
fi

mkdir -p "$repo_dir/results/mnist"
run_dir=$(mktemp -d "$repo_dir/results/mnist/run-XXXXXXXX")
mkdir "$run_dir/checkpoints"
for ((seed = 1; seed <= runs; seed++)); do
    mkdir "$run_dir/seed-$seed"
done

cd "$repo_dir"
runner=$repo_dir/experiments/mnist/run.slurm
array=(--array="1-$runs" --kill-on-invalid-dep=yes)
resources=(--nodes=1 --ntasks=1 --chdir="$repo_dir")

submit() {
    local job_id
    job_id=$(sbatch --parsable "$@")
    printf '%s\n' "${job_id%%;*}"
}

digital=$(submit "${array[@]}" "${resources[@]}" \
    --job-name=mnist-digital --partition="$gpu_partition" --gres=gpu:1 \
    --cpus-per-task=4 --mem=16G --time="$gpu_time" --output="$run_dir/seed-%a/digital.log" \
    "$runner" digital "$run_dir" "$epochs")

# Each WPO seed can start on CPUs as soon as its own baseline finishes.
wpo=$(submit "${array[@]}" "${resources[@]}" \
    --dependency="aftercorr:$digital" --job-name=mnist-wpo \
    --partition="$cpu_partition" --cpus-per-task=64 --hint=nomultithread --mem="$wpo_mem" --time="$wpo_time" \
    --output="$run_dir/seed-%a/wpo.log" "$runner" wpo "$run_dir" "$hat_epochs")

# Finish every digital GPU job before allowing any OI GPU job to start.
oi=$(submit "${array[@]}" "${resources[@]}" \
    --dependency="afterok:$digital" --job-name=mnist-oi \
    --partition="$gpu_partition" --gres=gpu:1 --cpus-per-task=4 --mem=16G --time="$gpu_time" \
    --output="$run_dir/seed-%a/oi.log" "$runner" oi "$run_dir" "$oi_epochs")

# Checkpoints are only a handoff between stages; retain logs and histories.
cleanup=$(submit --dependency="afterany:$digital:$wpo:$oi" \
    --job-name=mnist-cleanup --partition="$cpu_partition" --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=128M --time=00:01:00 --chdir="$repo_dir" \
    --output="$run_dir/cleanup.log" "$runner" cleanup "$run_dir")

printf 'Digital: %s\nWPO: %s\nOI: %s\nCleanup: %s\nLogs and histories: %s\n' \
    "$digital" "$wpo" "$oi" "$cleanup" "$run_dir"
