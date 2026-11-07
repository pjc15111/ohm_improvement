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
gpu_time=${GPU_TIME:-02:00:00}
wpo_time=${WPO_TIME:-02:00:00}
noise_mode=${NOISE_MODE:-fresh}
wpo_mem=${WPO_MEM:-64G}

if [[ ! $runs =~ ^[1-9][0-9]*$ || ! $epochs =~ ^[1-9][0-9]*$ || ! $oi_epochs =~ ^[1-9][0-9]*$ ]]; then
    echo 'RUNS, DIGITAL_EPOCHS, and OI_EPOCHS must be positive integers' >&2
    exit 1
fi
if [[ $noise_mode != fixed && $noise_mode != fresh ]]; then
    echo 'NOISE_MODE must be fixed or fresh' >&2
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

# Finish all baseline training before HAT, then finish all HAT before any OI.
hat=$(submit "${array[@]}" "${resources[@]}" \
    --dependency="afterok:$digital" --job-name=mnist-hat \
    --partition="$gpu_partition" --gres=gpu:1 --cpus-per-task=4 --mem=16G --time="$gpu_time" \
    --output="$run_dir/seed-%a/hat.log" "$runner" hat "$run_dir" "$hat_epochs")

# CPU WPO can start as soon as its own shared HAT checkpoint is ready.
wpo=$(submit "${array[@]}" "${resources[@]}" \
    --dependency="aftercorr:$hat" --job-name=mnist-wpo \
    --partition="$cpu_partition" --cpus-per-task=64 --hint=nomultithread --mem="$wpo_mem" --time="$wpo_time" \
    --output="$run_dir/seed-%a/wpo.log" "$runner" wpo "$run_dir")

printf 'Digital: %s\nHAT: %s\nWPO: %s\n' "$digital" "$hat" "$wpo"
cleanup_dependencies="$digital:$hat:$wpo"
for initialization in digital hat; do
    for realizations in 4 16; do
        label="oi-$initialization-r$realizations"
        job=$(submit "${array[@]}" "${resources[@]}" \
            --dependency="afterok:$hat" --job-name="mnist-$label" \
            --partition="$gpu_partition" --gres=gpu:1 --cpus-per-task=4 --mem=16G --time="$gpu_time" \
            --output="$run_dir/seed-%a/$label.log" "$runner" oi "$run_dir" \
            "$oi_epochs" "$realizations" "$initialization" "$noise_mode")
        cleanup_dependencies+=":$job"
        printf '%s: %s\n' "$label" "$job"
    done
done

# Checkpoints are handoffs between stages; retain only logs and histories.
cleanup=$(submit --dependency="afterany:$cleanup_dependencies" \
    --job-name=mnist-cleanup --partition="$cpu_partition" --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=128M --time=00:01:00 --chdir="$repo_dir" \
    --output="$run_dir/cleanup.log" "$runner" cleanup "$run_dir")
printf 'Cleanup: %s\nLogs and histories: %s\n' "$cleanup" "$run_dir"
