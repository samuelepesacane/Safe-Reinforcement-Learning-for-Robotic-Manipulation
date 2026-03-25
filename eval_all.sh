#!/bin/bash
set -e

# Full evaluation: all algorithms x shield conditions x seeds x environments
# 72 conditions total: 4 algos x 2 shield x 3 seeds x 3 envs
# Results saved to results/eval_<env>_<algo>_<cond>_seed<n>/metrics.csv

# SafetyPointPush1-v0

echo "=== [Push] PPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/ppo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_ppo_shield_off_seed0

echo "=== [Push] PPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/ppo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_ppo_shield_off_seed1

echo "=== [Push] PPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/ppo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_ppo_shield_off_seed2

echo "=== [Push] PPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/ppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_ppo_shield_on_seed0

echo "=== [Push] PPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/ppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_ppo_shield_on_seed1

echo "=== [Push] PPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/ppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_ppo_shield_on_seed2

echo "=== [Push] SAC shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/sac_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_sac_shield_off_seed0

echo "=== [Push] SAC shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/sac_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_sac_shield_off_seed1

echo "=== [Push] SAC shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/sac_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_sac_shield_off_seed2

echo "=== [Push] SAC shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/sac_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_sac_shield_on_seed0

echo "=== [Push] SAC shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/sac_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_sac_shield_on_seed1

echo "=== [Push] SAC shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/sac_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_sac_shield_on_seed2

echo "=== [Push] RCPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/rcpo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_rcpo_shield_off_seed0

echo "=== [Push] RCPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/rcpo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_rcpo_shield_off_seed1

echo "=== [Push] RCPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/rcpo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_rcpo_shield_off_seed2

echo "=== [Push] RCPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/rcpo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_rcpo_shield_on_seed0

echo "=== [Push] RCPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/rcpo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_rcpo_shield_on_seed1

echo "=== [Push] RCPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/rcpo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_rcpo_shield_on_seed2

echo "=== [Push] LagPPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/lagppo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_lagppo_shield_off_seed0

echo "=== [Push] LagPPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/lagppo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_lagppo_shield_off_seed1

echo "=== [Push] LagPPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/lagppo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_lagppo_shield_off_seed2

echo "=== [Push] LagPPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/lagppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_lagppo_shield_on_seed0

echo "=== [Push] LagPPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/lagppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_lagppo_shield_on_seed1

echo "=== [Push] LagPPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/push/lagppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_push_lagppo_shield_on_seed2

# SafetyPointGoal1-v0

echo "=== [Goal] PPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/ppo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_ppo_shield_off_seed0

echo "=== [Goal] PPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/ppo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_ppo_shield_off_seed1

echo "=== [Goal] PPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/ppo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_ppo_shield_off_seed2

echo "=== [Goal] PPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/ppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_ppo_shield_on_seed0

echo "=== [Goal] PPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/ppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_ppo_shield_on_seed1

echo "=== [Goal] PPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/ppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_ppo_shield_on_seed2

echo "=== [Goal] SAC shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/sac_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_sac_shield_off_seed0

echo "=== [Goal] SAC shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/sac_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_sac_shield_off_seed1

echo "=== [Goal] SAC shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/sac_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_sac_shield_off_seed2

echo "=== [Goal] SAC shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/sac_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_sac_shield_on_seed0

echo "=== [Goal] SAC shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/sac_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_sac_shield_on_seed1

echo "=== [Goal] SAC shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/sac_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_sac_shield_on_seed2

echo "=== [Goal] RCPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/rcpo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_rcpo_shield_off_seed0

echo "=== [Goal] RCPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/rcpo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_rcpo_shield_off_seed1

echo "=== [Goal] RCPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/rcpo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_rcpo_shield_off_seed2

echo "=== [Goal] RCPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/rcpo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_rcpo_shield_on_seed0

echo "=== [Goal] RCPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/rcpo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_rcpo_shield_on_seed1

echo "=== [Goal] RCPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/rcpo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_rcpo_shield_on_seed2

echo "=== [Goal] LagPPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/lagppo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_lagppo_shield_off_seed0

echo "=== [Goal] LagPPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/lagppo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_lagppo_shield_off_seed1

echo "=== [Goal] LagPPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/lagppo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_lagppo_shield_off_seed2

echo "=== [Goal] LagPPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/lagppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_lagppo_shield_on_seed0

echo "=== [Goal] LagPPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/lagppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_lagppo_shield_on_seed1

echo "=== [Goal] LagPPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyPointGoal1-v0 \
  --model_path checkpoints/goal/lagppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_goal_lagppo_shield_on_seed2

# SafetyCarGoal1-v0

echo "=== [Car] PPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/ppo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_ppo_shield_off_seed0

echo "=== [Car] PPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/ppo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_ppo_shield_off_seed1

echo "=== [Car] PPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/ppo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_ppo_shield_off_seed2

echo "=== [Car] PPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/ppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_ppo_shield_on_seed0

echo "=== [Car] PPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/ppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_ppo_shield_on_seed1

echo "=== [Car] PPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/ppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_ppo_shield_on_seed2

echo "=== [Car] SAC shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/sac_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_sac_shield_off_seed0

echo "=== [Car] SAC shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/sac_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_sac_shield_off_seed1

echo "=== [Car] SAC shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/sac_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_sac_shield_off_seed2

echo "=== [Car] SAC shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/sac_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_sac_shield_on_seed0

echo "=== [Car] SAC shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/sac_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_sac_shield_on_seed1

echo "=== [Car] SAC shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/sac_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_sac_shield_on_seed2

echo "=== [Car] RCPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/rcpo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_rcpo_shield_off_seed0

echo "=== [Car] RCPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/rcpo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_rcpo_shield_off_seed1

echo "=== [Car] RCPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/rcpo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_rcpo_shield_off_seed2

echo "=== [Car] RCPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/rcpo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_rcpo_shield_on_seed0

echo "=== [Car] RCPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/rcpo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_rcpo_shield_on_seed1

echo "=== [Car] RCPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/rcpo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_rcpo_shield_on_seed2

echo "=== [Car] LagPPO shield_off seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/lagppo_shield_off/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_lagppo_shield_off_seed0

echo "=== [Car] LagPPO shield_off seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/lagppo_shield_off/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_lagppo_shield_off_seed1

echo "=== [Car] LagPPO shield_off seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/lagppo_shield_off/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_lagppo_shield_off_seed2

echo "=== [Car] LagPPO shield_on seed 0 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/lagppo_shield_on/seed_0/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_lagppo_shield_on_seed0

echo "=== [Car] LagPPO shield_on seed 1 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/lagppo_shield_on/seed_1/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_lagppo_shield_on_seed1

echo "=== [Car] LagPPO shield_on seed 2 ===" && python -m src.evaluate \
  --env_id SafetyCarGoal1-v0 \
  --model_path checkpoints/car/lagppo_shield_on/seed_2/latest \
  --episodes 20 --seed 100 --log_dir results/eval_car_lagppo_shield_on_seed2

echo "=== All evaluations complete ==="
