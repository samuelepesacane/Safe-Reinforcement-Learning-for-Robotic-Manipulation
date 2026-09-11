#!/bin/bash
#
# Riemannian shield, post D1-D3 fix -- seeded runs.
#
# Question this answers: how does the (now-correct) Riemannian shield behave on
# a holonomic point robot (SafetyPointGoal1-v0) versus a nonholonomic car
# (SafetyCarGoal1-v0), under LagPPO. Prior Riemannian logs (logs/car_riemannian_lagppo_seed*,
# logs/goal_riemannian_lagppo_seed*) are contaminated by D1 (position read from
# accelerometer, not env position), D2 (hazards frozen at episode 0), and D3
# (gradient deflection sign flipped, clip saturated to a fixed 45-degree nudge)
# and must not be used as a comparator -- see SESSION_HANDOFF.md section 8.
#
# shield_alpha=0.1, shield_influence_radius=0.4 chosen from a fresh post-fix
# scan (SESSION_HANDOFF.md section 8, "Influence_radius re-scan"), on behavior
# (mid-range, non-saturated intervention rate), not by matching the old
# (contaminated) config -- 0.4 also being the old value there is coincidence.
#
# --eval_freq 20000 matches the existing shield-off logs' logging cadence so
# the lambda and intervention-rate series are directly comparable within seed.
#
# The existing LagPPO shield-off logs on both envs (logs/car_lagppo_shield_off_seed{0,1,2},
# logs/goal_lagppo_shield_off_seed{0,1,2}, confirmed present) are the free
# within-seed baseline for this comparison: shield-off is untouched by D1-D3,
# and a seed fixes the hazard layout, so seed k is the same geometry in both
# conditions.
#
# At 3 seeds this is a mechanism-verification run, not a results run: the
# two-sided permutation floor is p=0.10 and the paired sign-test floor is 0.25,
# and lambda does not bind within 1M steps at these hyperparameters (D5). Report
# per-seed values and paired point-vs-car contrasts; do not write this up as a
# significance claim.
#
# Runtime: ~45 min per run on the RTX 3070 (num_envs=4) => ~4.5 h for all 6.
#
# Usage:
#   nohup bash run_riemannian_postfix.sh > logs/run_riemannian_postfix.log 2>&1 &
#
# The script is resumable: any run whose checkpoint already exists is skipped,
# so an interrupted run can be restarted with the same command.

set -e

STEPS=1000000
BUDGET=0.05
LR_LAMBDA=5e-4
ENVS=4
EVAL_FREQ=20000
ALPHA=0.1
RADIUS=0.4
SEEDS="0 1 2"

declare -A ENV_IDS=( ["car"]="SafetyCarGoal1-v0" ["goal"]="SafetyPointGoal1-v0" )

for TAG_ENV in car goal; do
  ENV_ID="${ENV_IDS[$TAG_ENV]}"
  for SEED in $SEEDS; do
    LOG_DIR="logs/${TAG_ENV}_riemannian_postfix_lagppo_seed${SEED}"
    CKPT_DIR="checkpoints/${TAG_ENV}_riemannian_postfix/lagppo_shield_on/seed_${SEED}"

    if [ -f "${CKPT_DIR}/latest.zip" ]; then
      echo "=== SKIP (already done): env=${ENV_ID} seed=${SEED} ==="
      continue
    fi

    echo "=== TRAIN env=${ENV_ID} seed=${SEED} ===" && python -m src.train \
      --env_id "$ENV_ID" \
      --algo lagppo --total_timesteps $STEPS \
      --seed $SEED --num_envs $ENVS --eval_freq $EVAL_FREQ \
      --cost_budget $BUDGET --lr_lambda $LR_LAMBDA \
      --use_shield --shield_type riemannian \
      --shield_alpha $ALPHA --shield_influence_radius $RADIUS \
      --log_dir "$LOG_DIR" \
      --ckpt_dir "$CKPT_DIR"
  done
done

# Evaluation: shield off, no reward shaping (src.evaluate always builds a raw
# env by design), 20 episodes, seed 100 -- matching the protocol used for
# every other reported eval number. Training logs alone give lambda and the
# shield's own diagnostics but not cost/CVaR/violation-rate on the converged
# policy, which only src.evaluate computes.
#
# New results/ dirs (eval_{car,goal}_riemannian_postfix_lagppo_seed*),
# deliberately not eval_{car,goal}_riemannian_lagppo_seed* -- those are the
# pre-fix contaminated evals and must not be overwritten.
for TAG_ENV in car goal; do
  ENV_ID="${ENV_IDS[$TAG_ENV]}"
  for SEED in $SEEDS; do
    CKPT_DIR="checkpoints/${TAG_ENV}_riemannian_postfix/lagppo_shield_on/seed_${SEED}"
    RESULTS_DIR="results/eval_${TAG_ENV}_riemannian_postfix_lagppo_seed${SEED}"

    if [ -f "${RESULTS_DIR}/metrics.csv" ]; then
      echo "=== SKIP EVAL (already done): env=${ENV_ID} seed=${SEED} ==="
      continue
    fi

    echo "=== EVAL env=${ENV_ID} seed=${SEED} ===" && python -m src.evaluate \
      --env_id "$ENV_ID" \
      --model_path "${CKPT_DIR}/latest" \
      --episodes 20 --seed 100 \
      --log_dir "$RESULTS_DIR"
  done
done

echo "=== DONE. Read out with: python -m src.analysis.settling 'logs/*_riemannian_postfix_lagppo_seed*' ==="
