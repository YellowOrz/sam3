#!/usr/bin/env bash
# Run a frozen epoch checkpoint on a shared GPU; launch inside tmux for persistence.
set -euo pipefail

project=/home/zhengyuxi/projects/sam3-yelloworz
python=/home/zhengyuxi/.conda/envs/sam3-tokens/bin/python
dataset=/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1
base=/home/zhengyuxi/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
epoch=${EVAL_EPOCH:-1}
gpu=${EVAL_GPU:-1}
output=${EVAL_OUTPUT:?Set EVAL_OUTPUT to a new result directory}
checkpoint="$dataset/checkpoints/formal-k4-e2/k4_epoch${epoch}_complete.pt"

cd "$project"
test -f "$checkpoint"
# Refuse to overwrite earlier evaluation results.
mkdir "$output"
cp "$checkpoint" "$output/token_snapshot.pt"
cp scripts/evaluate_bilateral_tokens.py "$output/evaluator_snapshot.py"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$project:$project/scripts"
printf 'started_at=%s gpu=%s epoch=%s\n' "$(date --iso-8601=seconds)" "$gpu" "$epoch" > "$output/evaluation.log"
sha256sum "$checkpoint" "$output/token_snapshot.pt" "$output/evaluator_snapshot.py" >> "$output/evaluation.log"

# This bounds this evaluation run to two hours. The allocator cap excludes CUDA
# context/driver overhead; inspect nvidia-smi free memory before launching.
exec timeout --signal=TERM --kill-after=30s 7200s "$python" -u "$output/evaluator_snapshot.py" \
  --data-root "$dataset/val" \
  --base-checkpoint "$base" \
  --learned-checkpoint "epoch${epoch}=$output/token_snapshot.pt" \
  --include-ve \
  --output-dir "$output" \
  --batch-size 1 \
  --gpu-memory-fraction 0.25 \
  --samples-per-group 0 \
  --render-count-per-group 4 >> "$output/evaluation.log" 2>&1
