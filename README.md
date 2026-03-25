# Safe Reinforcement Learning for Robotic Manipulation

**Paper:** Geometric Shielding and Lagrangian Constraints are Complementary: An Empirical Study in Safe Reinforcement Learning

**Branch:** `riemannian-shield-extended` - extended study with three environments and a Riemannian shield. See `main` for the original single-environment paper code.

This repository contains the code for the ablation study described in the paper above, extended with two additional environments and a new shield type inspired by Jaquier et al. (IROS 2023).

Work still in progrss.

## What is new in this branch

The original paper studied four safe RL algorithms with and without a geometric keepout shield on a single environment (SafetyPointPush1-v0). This branch extends that study in two directions.

**Three environments** instead of one. The ablation now covers SafetyPointPush1-v0 (2 hazards, point robot), SafetyPointGoal1-v0 (8 hazards, point robot), and SafetyCarGoal1-v0 (8 hazards, car robot). This reveals that the geometric shield's effectiveness depends on environment structure: it works well in sparse hazard environments but degrades in denser ones.

**Riemannian shield** as a third shield condition for LagPPO. Inspired by the region-avoiding Riemannian metric construction in Jaquier et al. (IROS 2023, arXiv:2307.15440), this shield computes the gradient of an inverse-square barrier potential summed over all hazards simultaneously and subtracts it from the policy's proposed action at each step. This is a first-order approximation of the geodesic deflection that would arise under Jaquier et al.'s modified metric. It handles dense hazard fields more gracefully than the geometric shield because it reasons about the global hazard geometry rather than the nearest single hazard. Implementation: `src/safety/riemannian_shield.py`.

## Key findings from the extended study

The complementarity finding from the original paper holds and generalises, but with an important qualification: the shield must be matched to the environment's geometric complexity.

LagPPO seed-averaged eval results (per-step cost / CVaR at α=0.1):

| Environment | No shield | Geometric shield | Riemannian shield |
|---|---|---|---|
| Push (2 hazards, point) | 0.050 / 314.8 | 0.035 / 285.8 | 0.024 / 184.4 |
| Goal (8 hazards, point) | 0.046 / 95.2 | 0.052 / 107.5 | 0.048 / 115.3 |
| Car (8 hazards, car) | 0.064 / 178.3 | 0.061 / 151.2 | 0.051 / 121.3 |

The geometric shield degrades on PointGoal because its bisection logic handles one hazard at a time: in a dense field it can push the agent away from one hazard and toward another. The Riemannian shield partially recovers this degradation because the global barrier gradient accounts for all hazards simultaneously. The largest gains are on CarGoal, where the car's nonholonomic dynamics expose a mismatch between action-space correction and configuration-space constraint geometry.

An open question motivating further work: the Lagrange multiplier trajectory on CarGoal is noisier with the Riemannian shield than expected, suggesting that correcting actions after the policy proposes them is a different problem from shaping the geometry in which the policy learns. This is the gap between a first-order approximation and a true geodesic planner.

## Project status

This is a research prototype on a single 8GB-GPU machine. The focus is understanding how different safety mechanisms interact, not squeezing out the best possible score.

RCPO is implemented as a fixed-penalty baseline rather than the full multi-timescale algorithm from Tessler et al. (2018).

## Repository layout

```
src/
  train.py              main training entry point
  evaluate.py           offline evaluation, writes metrics.csv
  envs/make_env.py      environment factory + shield wiring
  algos/lagppo.py       Lagrangian PPO (LagrangianState, callback)
  safety/shield.py      geometric keepout shield (bisection-based)
  safety/riemannian_shield.py   Riemannian barrier shield (new)
  safety/metrics.py     CVaR and aggregation helpers
  callbacks/train_logging_callback.py   logs shield_intervention_rate
  plot_three_way.py     generates the three-way comparison figures

run_full_ablation.sh          original 24-condition Push ablation
run_ablation_pointgoal.sh     24-condition PointGoal ablation
run_ablation_cargoal.sh       24-condition CarGoal ablation
run_riemannian_full.sh        Riemannian shield training + eval pipeline
eval_all.sh                   full evaluation across all conditions
```

Checkpoints are organised as `checkpoints/{push,goal,car}/{algo}_{shield_off,shield_on}/seed_{0,1,2}/latest.zip` and `checkpoints/{push,goal,car}_riemannian/lagppo_shield_on/seed_{0,1,2}/latest.zip`.

## Supported environments

SafetyPointPush1-v0, SafetyPointGoal1-v0, SafetyCarGoal1-v0 from [Safety-Gymnasium](https://github.com/PKU-Alignment/Safety-Gymnasium).

## Shield types

Pass `--use_shield --shield_type geometric` for the original bisection-based keepout shield. Pass `--use_shield --shield_type riemannian` for the Riemannian barrier shield. Both accept the same training pipeline without modification.

Riemannian shield hyperparameters: `--shield_alpha` (gradient scale, default 0.1) and `--shield_influence_radius` (active zone beyond hazard boundary, default 0.5; used 0.4 in all reported experiments).

## Quickstart

```bash
# Install dependencies
conda create -n .venv python=3.10 -y
conda activate .venv
pip install -r requirements.txt

# Run a minimal LagPPO + Riemannian shield test (5k steps)
python -m src.train \
  --env_id SafetyPointGoal1-v0 \
  --algo lagppo --total_timesteps 5000 \
  --seed 0 --num_envs 4 --eval_freq 5000 \
  --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --shield_type riemannian \
  --shield_alpha 0.1 --shield_influence_radius 0.4 \
  --log_dir logs/test_riemannian \
  --ckpt_dir checkpoints/test_riemannian
```

## Reproducing the extended study

Run the original Push ablation (24 conditions):

```bash
nohup bash run_full_ablation.sh > logs/run_full_ablation.log 2>&1 &
```

Run the PointGoal and CarGoal ablations (24 conditions each):

```bash
nohup bash run_ablation_pointgoal.sh > logs/run_pointgoal.log 2>&1 &
nohup bash run_ablation_cargoal.sh > logs/run_cargoal.log 2>&1 &
```

Run the Riemannian shield training and evaluation pipeline:

```bash
nohup bash run_riemannian_full.sh > logs/run_riemannian_full.log 2>&1 &
```

Evaluate all conditions:

```bash
bash eval_all.sh 2>&1 | tee logs/eval_all.log
```

Generate the three-way comparison plots:

```bash
python -m src.plot_three_way
```

Plots are saved to `results/plots_three_way/`.

## System specifications

All experiments were run on a single consumer GPU machine.

- CPU: AMD Ryzen 7 3700X (8 cores, 16 threads)
- GPU: RTX 3070 (8GB VRAM)
- RAM: 32 GB DDR4
- OS: Ubuntu 24.04 on WSL2

## Background reading

- Altman, *Constrained Markov Decision Processes*, 1999
- Tessler et al., *Reward Constrained Policy Optimization*, arXiv:1805.11074
- Ray et al., *Benchmarking Safe Exploration in Deep RL*, 2019
- Ji et al., *Safety-Gymnasium*, arXiv:2310.12567
- Klein, Jaquier, Meixner, Asfour, *On the Design of Region-Avoiding Metrics for Collision-Safe Motion Generation on Riemannian Manifolds*, IROS 2023, arXiv:2307.15440

## Citation

```
Pesacane, S. (2026). Geometric Shielding and Lagrangian Constraints are Complementary:
An Empirical Study in Safe Reinforcement Learning.
https://github.com/samuelepesacane/Safe-Reinforcement-Learning-for-Robotic-Manipulation/
```

## License

MIT License (see `LICENSE`).

## AI assistance

The writing and documentation in this repository were edited with the assistance of an AI language model. All experimental design, implementation, results, and scientific conclusions are the authors' own.
