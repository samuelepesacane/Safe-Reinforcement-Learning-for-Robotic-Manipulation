#!/usr/bin/env bash
# Environment variables:
#   SEEDS       - Space-separated list of random seeds (default: "0 1 2")
#   TIMESTEPS   - Training timesteps per run (default: 150000)
#   OUT_CSV     - Output CSV for ablation summary (default: results/ablations_summary.csv)
#   PARALLEL    - Number of concurrent training jobs (default: 1 => sequential)
#
# Example:
#   SEEDS="0 1 2 3 4" TIMESTEPS=500000 OUT_CSV="results/custom_summary.csv" PARALLEL=3 bash scripts/train_all.sh

set -euo pipefail

# --- Configurable defaults ---
SEEDS="${SEEDS:-"0 1 2"}"
TIMESTEPS="${TIMESTEPS:-150000}"
OUT_CSV="${OUT_CSV:-results/ablations_summary.csv}"
PARALLEL="${PARALLEL:-1}"

# --- Create results folder if missing ---
mkdir -p results

echo "[Ablations] Starting ablation suite"
echo "[Ablations] Seeds      : $SEEDS"
echo "[Ablations] Timesteps  : $TIMESTEPS"
echo "[Ablations] Output CSV : $OUT_CSV"
echo "[Ablations] Parallel   : $PARALLEL"

# Run the ablation driver (this launches src.train and src.evaluate internally)
python -m src.ablations \
  --seeds $SEEDS \
  --total_timesteps $TIMESTEPS \
  --out_csv "$OUT_CSV" \
  --parallel $PARALLEL

# Plot aggregated results
echo "[Ablations] Generating plots..."
python -m src.visualize \
  --ablations_csv "$OUT_CSV" \
  --out_dir results/plots

echo "[Ablations] All experiments finished successfully."
