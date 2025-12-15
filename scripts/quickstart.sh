#!/usr/bin/env bash
set -e

echo "[Quickstart] Installing dependencies..."
pip install -r requirements.txt

echo "[Quickstart] Training LagPPO on SafetyPointPush1-v0 (100k steps)..."
python -m src.train \
  --env_id SafetyPointPush1-v0 \
  --algo lagppo \
  --total_timesteps 100000 \
  --seed 0 \
  --cost_budget 0.05 \
  --lr_lambda 5e-4 \
  --use_shield \
  --num_envs 4 \
  --eval_freq 10000 \
  --log_dir logs/quickstart_pointpush

echo "[Quickstart] Evaluating trained policy..."
python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/SafetyPointPush1-v0/lagppo/seed_0/latest \
  --episodes 10 \
  --seed 123 \
  --log_dir results/quickstart_eval

echo "[Quickstart] Visualizing learning curves..."
python -m src.visualize \
  --log_dirs logs/quickstart_pointpush \
  --out_dir results/quickstart_plots

echo "[Quickstart] Done."
