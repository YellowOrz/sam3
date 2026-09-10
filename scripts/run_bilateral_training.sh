#!/usr/bin/env bash
set -euo pipefail

project=/home/zhengyuxi/projects/sam3-yelloworz
python=/home/zhengyuxi/.conda/envs/sam3-tokens/bin/python
data=/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/train
base=/home/zhengyuxi/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
output=/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/checkpoints/formal-k4-e2
log="$output/training.log"

cd "$project"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

args=(
  --data-root "$data"
  --base-checkpoint "$base"
  --output-dir "$output"
  --tokens-per-class 4
  --batch-size 2
  --amp
  --epochs 2
  --learning-rate 0.01
  --seed 123
  --log-every 50
  --save-every 250
)

latest="$output/k4_epoch2_latest.pt"
if [[ -f "$latest" ]]; then
  args+=(--resume "$latest")
fi

printf 'started_at=%s\n' "$(date --iso-8601=seconds)" >> "$log"
printf 'command=%q ' "$python" scripts/train_learnable_tokens.py "${args[@]}" >> "$log"
printf '\n' >> "$log"
exec "$python" scripts/train_learnable_tokens.py "${args[@]}" >> "$log" 2>&1
