#!/bin/bash
#
# Run 2a -- dual-authority control on SafetyCarGoal1-v0.
#
# Question this answers: is the car "multiplier never settles" anomaly simply an
# open dual loop? The audit (CODE_AUDIT_AND_EXECUTION_PLAN.md, finding D5) showed
# that lambda never exceeds 0.0075 in any existing run, while the per-step reward
# on the car is ~0.029 and the cost rate ~0.064. For the penalty lambda*c to be
# comparable to the reward signal you need lambda ~ 0.45 -- about 100x more than
# observed. This script raises the dual gain by exactly that factor and asks
# whether the multiplier then settles.
#
# IMPORTANT: every run here is SHIELD-OFF. That is deliberate, not an oversight.
# The shield defects D1-D4 (position read from the wrong observation indices,
# hazards frozen at episode 0, Riemannian sign error) make any shielded result
# uninterpretable until Run 0 lands. Shield-off is untouched by all of them, so
# this experiment is valid on the current code with no changes.
#
# No source changes are required: --lr_lambda already exists (train.py:79-84)
# and flows into LagrangianState at train.py:507-512.
#
# Design: 8 seeds x 2 dual gains = 16 runs.
#   lr_lambda = 5e-4  -- the paper's value, the matched control arm
#   lr_lambda = 5e-2  -- 100x, the dual-authority treatment
#
# Seeds 0-2 at 5e-4 duplicate the March runs in logs/car_lagppo_shield_off_seed*.
# Re-running them is intentional: it gives a like-for-like comparator produced by
# the same code at the same time, and doubles as a reproducibility check against
# the old logs. If they do NOT reproduce, that is itself a finding worth knowing
# before anything else is built on top of them.
#
# Runtime: ~45 min per run on the RTX 3070 => ~12 h for all 16. One overnight.
#
# Usage:
#   nohup bash run_dualgain_car.sh > logs/run_dualgain_car.log 2>&1 &
#
# The script is resumable: any run whose checkpoint already exists is skipped,
# so an interrupted overnight can be restarted with the same command.

set -e

ENV="SafetyCarGoal1-v0"
STEPS=1000000
BUDGET=0.05
ENVS=4
EVAL_FREQ=20000
SEEDS="0 1 2 3 4 5 6 7"
GAINS="5e-4 5e-2"

for GAIN in $GAINS; do
  for SEED in $SEEDS; do
    TAG="car_off_lr${GAIN}_seed${SEED}"
    LOG_DIR="logs/dualgain_${TAG}"
    CKPT_DIR="checkpoints/dualgain/car_lagppo_off_lr${GAIN}/seed_${SEED}"

    if [ -f "${CKPT_DIR}/latest.zip" ]; then
      echo "=== SKIP (already done): lr_lambda=${GAIN} seed=${SEED} ==="
      continue
    fi

    echo "=== TRAIN lr_lambda=${GAIN} seed=${SEED} ==="
    python -m src.train \
      --env_id $ENV --algo lagppo --total_timesteps $STEPS \
      --seed $SEED --num_envs $ENVS --eval_freq $EVAL_FREQ \
      --cost_budget $BUDGET --lr_lambda $GAIN \
      --log_dir "$LOG_DIR" \
      --ckpt_dir "$CKPT_DIR"
  done
done

# Evaluation: shield off and no reward shaping, matching the protocol used for
# every other reported number (evaluate.py builds a raw env; 20 episodes, seed 100).
for GAIN in $GAINS; do
  for SEED in $SEEDS; do
    TAG="car_off_lr${GAIN}_seed${SEED}"
    echo "=== EVAL lr_lambda=${GAIN} seed=${SEED} ==="
    python -m src.evaluate \
      --env_id $ENV \
      --model_path "checkpoints/dualgain/car_lagppo_off_lr${GAIN}/seed_${SEED}/latest" \
      --episodes 20 --seed 100 \
      --log_dir "results/eval_dualgain_${TAG}"
  done
done

echo "=== DONE. Read out with: python -m src.analysis.settling logs/dualgain_* ==="
