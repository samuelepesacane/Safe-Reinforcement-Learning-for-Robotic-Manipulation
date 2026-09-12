#!/bin/bash
#
# Causal test of the shield's kinematic prediction model (SESSION_HANDOFF.md,
# measurement B / the fit_kinematic_model.py result): does replacing the
# world-frame xy-velocity assumption with the data-fit heading-relative model
# (kinematic_model="heading_fit", src/safety/kinematic_fits.py) change entry
# prevention / dwell time relative to the current model (kinematic_model=
# "world_xy", default, unchanged behavior)?
#
# Short-horizon (150k steps, not the full 1M) mechanism-verification run, 1
# seed, both envs, BOTH kinematic models trained at the same horizon so the
# comparison is apples-to-apples (not against the existing full-1M
# riemannian_postfix numbers, which ran a different horizon and only the old
# model). Those 1M-step numbers (car 0.95-1.15 entries/episode, 654-810 dwell;
# point 3.15-3.55 entries/episode, 15-19 dwell) remain useful CONTEXT but are
# not the primary comparator here.
#
# Runtime: ~45min/1M steps (RTX 3070, num_envs=4) => ~7 min/run => ~28 min
# for all 4 training runs.
#
# Usage:
#   nohup bash run_kinematic_model_causal_test.sh > logs/run_kinematic_model_causal_test.log 2>&1 &
#
# Resumable: skips any run whose checkpoint already exists.

set -e

STEPS=150000
BUDGET=0.05
LR_LAMBDA=5e-4
ENVS=4
EVAL_FREQ=10000
ALPHA=0.1
RADIUS=0.4
SEED=0

declare -A ENV_IDS=( ["car"]="SafetyCarGoal1-v0" ["goal"]="SafetyPointGoal1-v0" )

for TAG_ENV in car goal; do
  ENV_ID="${ENV_IDS[$TAG_ENV]}"
  for MODEL in world_xy heading_fit; do
    LOG_DIR="logs/${TAG_ENV}_kinmodel_${MODEL}_lagppo_seed${SEED}"
    CKPT_DIR="checkpoints/${TAG_ENV}_kinmodel_${MODEL}/lagppo_shield_on/seed_${SEED}"

    if [ -f "${CKPT_DIR}/latest.zip" ]; then
      echo "=== SKIP (already done): env=${ENV_ID} model=${MODEL} ==="
      continue
    fi

    echo "=== TRAIN env=${ENV_ID} model=${MODEL} ===" && .venv/bin/python -m src.train \
      --env_id "$ENV_ID" \
      --algo lagppo --total_timesteps $STEPS \
      --seed $SEED --num_envs $ENVS --eval_freq $EVAL_FREQ \
      --cost_budget $BUDGET --lr_lambda $LR_LAMBDA \
      --use_shield --shield_type riemannian \
      --shield_alpha $ALPHA --shield_influence_radius $RADIUS \
      --shield_kinematic_model $MODEL \
      --log_dir "$LOG_DIR" \
      --ckpt_dir "$CKPT_DIR"
  done
done

echo "=== DONE training. Run scripts/entry_dwell_probe.py against the 4 checkpoints next. ==="
