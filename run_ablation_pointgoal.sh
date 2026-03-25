#!/bin/bash
set -e

# Full 4-algorithm x 2-shield ablation on SafetyPointGoal1-v0
# 24 runs total: 4 algos x 2 conditions x 3 seeds

ENV="SafetyPointGoal1-v0"
STEPS=1000000
BUDGET=0.05
LR_LAM=5e-4
PENALTY=1.0
ENVS=4
EVAL_FREQ=20000

# --- PPO shield OFF ---
echo "=== PPO shield_off seed 0 ===" && python -m src.train \
  --env_id $ENV --algo ppo --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --log_dir logs/goal_ppo_shield_off_seed0 \
  --ckpt_dir checkpoints/goal/ppo_shield_off/seed_0

echo "=== PPO shield_off seed 1 ===" && python -m src.train \
  --env_id $ENV --algo ppo --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --log_dir logs/goal_ppo_shield_off_seed1 \
  --ckpt_dir checkpoints/goal/ppo_shield_off/seed_1

echo "=== PPO shield_off seed 2 ===" && python -m src.train \
  --env_id $ENV --algo ppo --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --log_dir logs/goal_ppo_shield_off_seed2 \
  --ckpt_dir checkpoints/goal/ppo_shield_off/seed_2

# --- PPO shield ON ---
echo "=== PPO shield_on seed 0 ===" && python -m src.train \
  --env_id $ENV --algo ppo --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --use_shield \
  --log_dir logs/goal_ppo_shield_on_seed0 \
  --ckpt_dir checkpoints/goal/ppo_shield_on/seed_0

echo "=== PPO shield_on seed 1 ===" && python -m src.train \
  --env_id $ENV --algo ppo --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --use_shield \
  --log_dir logs/goal_ppo_shield_on_seed1 \
  --ckpt_dir checkpoints/goal/ppo_shield_on/seed_1

echo "=== PPO shield_on seed 2 ===" && python -m src.train \
  --env_id $ENV --algo ppo --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --use_shield \
  --log_dir logs/goal_ppo_shield_on_seed2 \
  --ckpt_dir checkpoints/goal/ppo_shield_on/seed_2

# --- SAC shield OFF ---
echo "=== SAC shield_off seed 0 ===" && python -m src.train \
  --env_id $ENV --algo sac --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --log_dir logs/goal_sac_shield_off_seed0 \
  --ckpt_dir checkpoints/goal/sac_shield_off/seed_0

echo "=== SAC shield_off seed 1 ===" && python -m src.train \
  --env_id $ENV --algo sac --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --log_dir logs/goal_sac_shield_off_seed1 \
  --ckpt_dir checkpoints/goal/sac_shield_off/seed_1

echo "=== SAC shield_off seed 2 ===" && python -m src.train \
  --env_id $ENV --algo sac --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --log_dir logs/goal_sac_shield_off_seed2 \
  --ckpt_dir checkpoints/goal/sac_shield_off/seed_2

# --- SAC shield ON ---
echo "=== SAC shield_on seed 0 ===" && python -m src.train \
  --env_id $ENV --algo sac --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --use_shield \
  --log_dir logs/goal_sac_shield_on_seed0 \
  --ckpt_dir checkpoints/goal/sac_shield_on/seed_0

echo "=== SAC shield_on seed 1 ===" && python -m src.train \
  --env_id $ENV --algo sac --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --use_shield \
  --log_dir logs/goal_sac_shield_on_seed1 \
  --ckpt_dir checkpoints/goal/sac_shield_on/seed_1

echo "=== SAC shield_on seed 2 ===" && python -m src.train \
  --env_id $ENV --algo sac --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --use_shield \
  --log_dir logs/goal_sac_shield_on_seed2 \
  --ckpt_dir checkpoints/goal/sac_shield_on/seed_2

# --- RCPO shield OFF ---
echo "=== RCPO shield_off seed 0 ===" && python -m src.train \
  --env_id $ENV --algo rcpo --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --penalty_coef $PENALTY \
  --log_dir logs/goal_rcpo_shield_off_seed0 \
  --ckpt_dir checkpoints/goal/rcpo_shield_off/seed_0

echo "=== RCPO shield_off seed 1 ===" && python -m src.train \
  --env_id $ENV --algo rcpo --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --penalty_coef $PENALTY \
  --log_dir logs/goal_rcpo_shield_off_seed1 \
  --ckpt_dir checkpoints/goal/rcpo_shield_off/seed_1

echo "=== RCPO shield_off seed 2 ===" && python -m src.train \
  --env_id $ENV --algo rcpo --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --penalty_coef $PENALTY \
  --log_dir logs/goal_rcpo_shield_off_seed2 \
  --ckpt_dir checkpoints/goal/rcpo_shield_off/seed_2

# --- RCPO shield ON ---
echo "=== RCPO shield_on seed 0 ===" && python -m src.train \
  --env_id $ENV --algo rcpo --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --penalty_coef $PENALTY --use_shield \
  --log_dir logs/goal_rcpo_shield_on_seed0 \
  --ckpt_dir checkpoints/goal/rcpo_shield_on/seed_0

echo "=== RCPO shield_on seed 1 ===" && python -m src.train \
  --env_id $ENV --algo rcpo --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --penalty_coef $PENALTY --use_shield \
  --log_dir logs/goal_rcpo_shield_on_seed1 \
  --ckpt_dir checkpoints/goal/rcpo_shield_on/seed_1

echo "=== RCPO shield_on seed 2 ===" && python -m src.train \
  --env_id $ENV --algo rcpo --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --penalty_coef $PENALTY --use_shield \
  --log_dir logs/goal_rcpo_shield_on_seed2 \
  --ckpt_dir checkpoints/goal/rcpo_shield_on/seed_2

# --- LagPPO shield OFF ---
echo "=== LagPPO shield_off seed 0 ===" && python -m src.train \
  --env_id $ENV --algo lagppo --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --cost_budget $BUDGET --lr_lambda $LR_LAM \
  --log_dir logs/goal_lagppo_shield_off_seed0 \
  --ckpt_dir checkpoints/goal/lagppo_shield_off/seed_0

echo "=== LagPPO shield_off seed 1 ===" && python -m src.train \
  --env_id $ENV --algo lagppo --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --cost_budget $BUDGET --lr_lambda $LR_LAM \
  --log_dir logs/goal_lagppo_shield_off_seed1 \
  --ckpt_dir checkpoints/goal/lagppo_shield_off/seed_1

echo "=== LagPPO shield_off seed 2 ===" && python -m src.train \
  --env_id $ENV --algo lagppo --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --cost_budget $BUDGET --lr_lambda $LR_LAM \
  --log_dir logs/goal_lagppo_shield_off_seed2 \
  --ckpt_dir checkpoints/goal/lagppo_shield_off/seed_2

# --- LagPPO shield ON ---
echo "=== LagPPO shield_on seed 0 ===" && python -m src.train \
  --env_id $ENV --algo lagppo --total_timesteps $STEPS \
  --seed 0 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --cost_budget $BUDGET --lr_lambda $LR_LAM --use_shield \
  --log_dir logs/goal_lagppo_shield_on_seed0 \
  --ckpt_dir checkpoints/goal/lagppo_shield_on/seed_0

echo "=== LagPPO shield_on seed 1 ===" && python -m src.train \
  --env_id $ENV --algo lagppo --total_timesteps $STEPS \
  --seed 1 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --cost_budget $BUDGET --lr_lambda $LR_LAM --use_shield \
  --log_dir logs/goal_lagppo_shield_on_seed1 \
  --ckpt_dir checkpoints/goal/lagppo_shield_on/seed_1

echo "=== LagPPO shield_on seed 2 ===" && python -m src.train \
  --env_id $ENV --algo lagppo --total_timesteps $STEPS \
  --seed 2 --num_envs $ENVS --eval_freq $EVAL_FREQ \
  --cost_budget $BUDGET --lr_lambda $LR_LAM --use_shield \
  --log_dir logs/goal_lagppo_shield_on_seed2 \
  --ckpt_dir checkpoints/goal/lagppo_shield_on/seed_2

echo "=== All SafetyPointGoal1-v0 runs complete ==="
