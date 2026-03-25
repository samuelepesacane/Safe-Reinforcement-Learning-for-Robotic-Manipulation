#!/bin/bash
set -e

# Complete Riemannian shield pipeline
# Phase 1: Training (5 runs + 3 Push negative control runs)
# Phase 2: Evaluation of all new checkpoints

# PHASE 1A: SafetyPointGoal1-v0 seeds 1 and 2
# (seed 0 already done before this script was launched)

echo "=== [Train] PointGoal Riemannian LagPPO seed 1 ===" && python -m src.train \
  --env_id SafetyPointGoal1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 1 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/goal_riemannian_lagppo_seed1 \
  --ckpt_dir checkpoints/goal_riemannian/lagppo_shield_on/seed_1

echo "=== [Train] PointGoal Riemannian LagPPO seed 2 ===" && python -m src.train \
  --env_id SafetyPointGoal1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 2 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/goal_riemannian_lagppo_seed2 \
  --ckpt_dir checkpoints/goal_riemannian/lagppo_shield_on/seed_2

# PHASE 1B: SafetyCarGoal1-v0 seeds 0, 1, 2

echo "=== [Train] CarGoal Riemannian LagPPO seed 0 ===" && python -m src.train \
  --env_id SafetyCarGoal1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 0 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/car_riemannian_lagppo_seed0 \
  --ckpt_dir checkpoints/car_riemannian/lagppo_shield_on/seed_0

echo "=== [Train] CarGoal Riemannian LagPPO seed 1 ===" && python -m src.train \
  --env_id SafetyCarGoal1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 1 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/car_riemannian_lagppo_seed1 \
  --ckpt_dir checkpoints/car_riemannian/lagppo_shield_on/seed_1

echo "=== [Train] CarGoal Riemannian LagPPO seed 2 ===" && python -m src.train \
  --env_id SafetyCarGoal1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 2 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/car_riemannian_lagppo_seed2 \
  --ckpt_dir checkpoints/car_riemannian/lagppo_shield_on/seed_2

# PHASE 1C: SafetyPointPush1-v0 seeds 0, 1, 2
# Negative control: geometric shield already works on Push
# (2 hazards, simple dynamics). If Riemannian performs similarly
# here, the PointGoal/CarGoal improvement is tied to env complexity.

echo "=== [Train] Push Riemannian LagPPO seed 0 ===" && python -m src.train \
  --env_id SafetyPointPush1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 0 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/push_riemannian_lagppo_seed0 \
  --ckpt_dir checkpoints/push_riemannian/lagppo_shield_on/seed_0

echo "=== [Train] Push Riemannian LagPPO seed 1 ===" && python -m src.train \
  --env_id SafetyPointPush1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 1 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/push_riemannian_lagppo_seed1 \
  --ckpt_dir checkpoints/push_riemannian/lagppo_shield_on/seed_1

echo "=== [Train] Push Riemannian LagPPO seed 2 ===" && python -m src.train \
  --env_id SafetyPointPush1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 2 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/push_riemannian_lagppo_seed2 \
  --ckpt_dir checkpoints/push_riemannian/lagppo_shield_on/seed_2

echo "=== All training complete. Starting evaluations. ==="

# PHASE 2: Evaluate all new Riemannian checkpoints

# PointGoal -- all three seeds (seed 0 evaluated separately earlier)
echo "=== [Eval] PointGoal Riemannian seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal_riemannian/lagppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_goal_riemannian_lagppo_seed0

echo "=== [Eval] PointGoal Riemannian seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal_riemannian/lagppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_goal_riemannian_lagppo_seed1

echo "=== [Eval] PointGoal Riemannian seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal_riemannian/lagppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_goal_riemannian_lagppo_seed2

# CarGoal -- all three seeds
echo "=== [Eval] CarGoal Riemannian seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car_riemannian/lagppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_car_riemannian_lagppo_seed0

echo "=== [Eval] CarGoal Riemannian seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car_riemannian/lagppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_car_riemannian_lagppo_seed1

echo "=== [Eval] CarGoal Riemannian seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car_riemannian/lagppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_car_riemannian_lagppo_seed2

# Push negative control -- all three seeds
echo "=== [Eval] Push Riemannian seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push_riemannian/lagppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_push_riemannian_lagppo_seed0

echo "=== [Eval] Push Riemannian seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push_riemannian/lagppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_push_riemannian_lagppo_seed1

echo "=== [Eval] Push Riemannian seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push_riemannian/lagppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 \
  --log_dir results/eval_push_riemannian_lagppo_seed2

echo "=== All training and evaluation complete ==="
