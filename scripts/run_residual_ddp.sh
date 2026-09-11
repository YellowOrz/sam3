#!/usr/bin/env bash
# Single-node, explicitly selected three/four-GPU residual training launcher.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash scripts/run_residual_ddp.sh --gpus 0,1,2 [--python /path/to/python] [--dry-run] -- TRAINER_ARGS...' \
        '' \
        'Select exactly three or four distinct physical GPU indices yourself.' \
        'Python defaults to SAM3_PYTHON, then python from PATH.' \
        'All arguments after -- go unchanged to the scripts.train_residual_ddp module.' \
        'The repository is prepended to PYTHONPATH; your working directory stays unchanged.' \
        'CUDA_DEVICE_ORDER=PCI_BUS_ID keeps physical indices aligned with nvidia-smi.' \
        'This launches one node with max_restarts=0; tmux/log capture stays outside.' \
        '--dry-run prints the quoted command without starting Python or using GPUs.'
}

fail() {
    printf 'run_residual_ddp.sh: %s\n' "$1" >&2
    exit 2
}

sam3_python="${SAM3_PYTHON:-python}"
sam3_gpus=''
sam3_dry_run=0
sam3_have_separator=0
while (($#)); do
    case "$1" in
        --gpus)
            (($# >= 2)) || fail '--gpus requires a comma-separated GPU list'
            [[ -z "$sam3_gpus" ]] || fail '--gpus may only be supplied once'
            sam3_gpus="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || fail '--python requires one executable path or name'
            [[ -n "$2" ]] || fail '--python cannot be empty'
            sam3_python="$2"
            shift 2
            ;;
        --dry-run)
            sam3_dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            sam3_have_separator=1
            shift
            break
            ;;
        *) fail "Unknown launcher option '$1'; put trainer arguments after --" ;;
    esac
done

[[ -n "$sam3_gpus" ]] || fail 'Explicit --gpus is required; GPUs are never chosen automatically'
[[ "$sam3_gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){2,3}$ ]] || \
    fail '--gpus must contain exactly three or four comma-separated nonnegative GPU indices'
IFS=',' read -r -a sam3_gpu_ids <<< "$sam3_gpus"
sam3_seen=','
for sam3_gpu_id in "${sam3_gpu_ids[@]}"; do
    [[ "$sam3_seen" != *",$sam3_gpu_id,"* ]] || fail 'GPU indices must be distinct'
    sam3_seen+="$sam3_gpu_id,"
done
((sam3_have_separator)) || fail 'Separate launcher options and trainer arguments with --'
(($#)) || fail 'Supply trainer arguments after --'
command -v -- "$sam3_python" >/dev/null 2>&1 || fail "Python executable not found: $sam3_python"
sam3_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
sam3_repo_root="$(cd -- "$sam3_launcher_dir/.." && pwd -P)"
sam3_trainer="$sam3_launcher_dir/train_residual_ddp.py"
[[ -f "$sam3_trainer" ]] || fail "Trainer not found next to launcher: $sam3_trainer"

sam3_command=(env "CUDA_VISIBLE_DEVICES=$sam3_gpus" CUDA_DEVICE_ORDER=PCI_BUS_ID
    "PYTHONPATH=$sam3_repo_root${PYTHONPATH:+:$PYTHONPATH}" "$sam3_python" -m torch.distributed.run
    --standalone --nnodes=1 "--nproc_per_node=${#sam3_gpu_ids[@]}" --max_restarts=0
    --module scripts.train_residual_ddp "$@")
if ((sam3_dry_run)); then
    printf '%q ' "${sam3_command[@]}"
    printf '\n'
    exit 0
fi
exec "${sam3_command[@]}"
