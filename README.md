# Safe Reinforcement Learning for Robotic Manipulation

**Paper:** [Geometric Shielding and Lagrangian Constraints are Complementary: An Empirical Study in Safe Reinforcement Learning](https://github.com/samuelepesacane/Safe-Reinforcement-Learning-for-Robotic-Manipulation/)

This repository contains the code for the ablation study described in the paper above.
It implements four safe RL algorithms (PPO, SAC, RCPO, LagPPO) with and without a geometric keepout shield, evaluated on the SafetyPointPush1-v0 benchmark from Safety-Gymnasium.

## Project status & context

This is a research prototype on a single 8GB-GPU machine.
The focus is:

- studying how Lagrangian CMDPs, fixed penalties (RCPO-style), and geometric shields trade off return vs safety
- keeping the code small and readable so it is easy to change and extend

RCPO is implemented as a fixed-penalty baseline rather than the full multi-timescale algorithm described in the original paper. It serves as a static penalty comparison point, not a faithful reproduction of Tessler et al. (2018).

## Experimental results

The main results of the 4$\times$2 factorial ablation (4 algorithms $\times$ shield on/off $\times$ 3 seeds $\times$ 1M steps) are reported in the paper. Summary:

| Condition | Avg Return | Avg Cost | Violation Rate | CVaR ($\alpha$=0.1) |
|---|---|---|---|---|
| PPO shield_off | 1.10 ± 0.47 | 53.4 ± 21.2 | 0.32 ± 0.12 | 375.8 |
| PPO shield_on | 0.68 ± 0.21 | 48.3 ± 45.5 | 0.25 ± 0.16 | 323.6 |
| SAC shield_off | 1.07 ± 0.11 | 21.2 ± 17.0 | 0.13 ± 0.00 | 200.5 |
| SAC shield_on | 0.95 ± 0.12 | 57.3 ± 56.0 | 0.15 ± 0.14 | 416.0 |
| RCPO shield_off | -0.90 ± 0.44 | 3.6 ± 3.8 | 0.08 ± 0.10 | 26.1 |
| RCPO shield_on | -0.26 ± 0.44 | 5.6 ± 8.1 | 0.03 ± 0.05 | 56.3 |
| LagPPO shield_off | 0.78 ± 0.23 | 49.8 ± 27.9 | 0.30 ± 0.15 | 314.8 |
| LagPPO shield_on | 0.84 ± 0.57 | 34.7 ± 28.6 | 0.22 ± 0.10 | 285.8 |

Key findings:
- RCPO achieves the lowest violation rate and CVaR but suppresses task performance almost entirely
- Geometric shielding and Lagrangian constraints are complementary: adding a shield measurably reduces Lagrange multiplier growth in LagPPO
- No algorithm achieves zero violations within 1M training steps

## Project Overview

This project investigates **safe reinforcement learning (Safe RL)** algorithms for robotic control tasks, with a focus on balancing reward maximization and constraint satisfaction. Multiple algorithms are implemented and evaluated:

- **Baseline algorithms**: PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic).
- **Constrained algorithms**: LagPPO (Lagrangian-based PPO), RCPO (fixed-penalty baseline).
- **Ablation studies**: systematic variation of shield condition across all four algorithms.

Experiments are run in **Safety-Gymnasium** environments (e.g. `SafetyPointPush1-v0`) where agents must achieve goals while minimizing violations of safety constraints during exploration.

## Motivation

Safe RL is one of the main bottlenecks for real robots: you want the agent to explore and learn, but you do not want it to collide with everything on the way.

This repo combines a few simple ideas:

- **Lagrangian CMDPs**: treat safety as a cost with a budget and update a Lagrange multiplier online.
- **Geometric shields**: a small 2D keep-out module that projects actions away from known hazards.

The point is not to squeeze out the best possible score, but to understand how these pieces interact under realistic compute constraints.

## Repository layout

- `src/train.py`: main training entry point (PPO, SAC, LagPPO, RCPO baseline)
- `src/evaluate.py`: offline evaluation on the true environment, writes `metrics.csv`
- `src/ablations.py`: small ablation launcher (wraps train + eval)
- `src/envs/make_env.py`: environment factory (Safety-Gymnasium) + shield wiring
- `src/algos/lagppo.py`: Lagrangian PPO (`LagrangianState`, callback)
- `src/safety/shield.py`: geometric keep-out shield for 2D point/car robots
- `src/safety/metrics.py`: helpers to aggregate returns/costs/violation rate/CVaR
- `scripts/quickstart.sh`: one small LagPPO + shield run on `SafetyPointPush1-v0`
- `run_full_ablation.sh`: runs all 24 conditions and saves logs

If you are trying to hack on something:
- **LagPPO / $\lambda$ updates** start from `src/algos/lagppo.py`
- **Shield**: `src/safety/shield.py` and `src/envs/make_env.py`

## Supported Environments

This project uses [**Safety-Gymnasium**](https://github.com/PKU-Alignment/Safety-Gymnasium), a benchmark library for safe reinforcement learning.
Unlike standard RL benchmarks that only optimize reward, Safety-Gymnasium introduces **safety costs** (e.g. entering hazard zones) and **constraint thresholds** that the agent should satisfy while learning.
Full documentation: [Safety-Gymnasium Docs](https://safety-gymnasium.readthedocs.io/en/latest/introduction/about_safety_gymnasium.html)

Main environments:
- **SafetyPointPush1-v0**: A point robot pushing a box to a goal while avoiding hazards (used in the paper)
- **SafetyPointButton1-v0**: A point robot navigating to a button with hazards in the way
- **SafetyCarPush1-v0**: A car-like robot pushing a box while avoiding hazards

These environments provide:
- **Reward** for doing the task (reach goal, push box, press button)
- **Cost** for unsafe events (stepping into hazard regions)
- **Budget** that algorithms like LagPPO or RCPO attempt to respect

## Gymnasium-Robotics Environments

In addition to Safety-Gymnasium, this project includes optional support for [**Gymnasium-Robotics**](https://gymnasium.farama.org/environments/robotics/), a set of continuous-control benchmarks for robotic manipulation tasks.

Unlike Safety-Gymnasium, **Gymnasium-Robotics environments do not include explicit safety costs**. They are useful for testing baseline performance without safety constraints.

### FetchPush-v2

- **Task**: A simulated Fetch robotic arm pushes a small box across a table to a target goal location
- **Observation space**: robot joint positions and velocities, gripper state, object positions
- **Action space**: 4D continuous control (three end-effector velocity commands + gripper open/close)
- **Reward function**: sparse or dense depending on configuration

> **Note:** FetchPush-v2 support is optional and has not been fully tested in this repository. A configuration file is provided at `configs/fetch_push.yaml`.

---

## Direct MuJoCo XML Integration

This section describes how to use a custom robot model directly with MuJoCo, bypassing Safety-Gymnasium.
You do not need this for the main ablation experiments described in the paper.
Skip to [Quickstart](#Quickstart) if you only want to reproduce the paper results.

---

Up to this point, every environment in this repo goes through **Safety-Gymnasium**.

That is the right choice for studying safe RL algorithms: Safety-Gymnasium gives you reward,
cost, hazard geometry, and a well-defined constraint budget out of the box.

But if you want to ask:
- *What if I have a different robot model?*
- *What if I want to define my own hazard zones from scratch?*
- *What if I want to test the pipeline on a model I designed?*

then you need to go one level below Safety-Gymnasium and talk to **MuJoCo directly**.

That is what `mujoco_connector.py` does.

---

### What MuJoCo actually is (and what Safety-Gymnasium hides from you)

MuJoCo is a physics engine.
You describe a robot and a world in an XML file, and MuJoCo simulates the physics.

When you call `gym.make("SafetyPointPush1-v0")`, Safety-Gymnasium loads a MuJoCo model internally, builds observations, applies reward and cost functions, and wraps everything in a standard Gymnasium interface.

The connector cuts those layers out. It loads any `.xml` file you give it, talks to `mujoco.MjModel` and `mujoco.MjData` directly, and builds a Gymnasium environment from scratch that the rest of this pipeline can use without modification.

---

### What the connector does

`mujoco_connector.py` defines one main class: `MujocoRoboticEnv`.

It is a standard `gymnasium.Env` that:

- loads any MuJoCo `.xml` model by path
- builds the **observation** as: joint positions ∥ joint velocities ∥ end-effector position ∥ goal position
- builds the **action** as: normalized torques in `[-1, 1]`, rescaled to each actuator's physical range
- computes **reward** as: negative distance from end-effector to goal, plus a success bonus
- computes **cost** as: penetration depth into user-defined hazard geometries, clipped to `[0, 1]`
- returns an **info dict** with `cost`, `is_success`, `shield_intervened`

---

### How the integration works

The natural place to add a new environment type is `src/envs/make_env.py`.

The patch is small. Five new lines go at the top of `_try_make()`, before the original `gym.make()` call:

```python
def _try_make(env_id: str, seed: int) -> gym.Env:
    if env_id.startswith("mujoco:"):
        model_path = env_id[len("mujoco:"):]
        env = MujocoRoboticEnv(model_path=model_path)
        env.reset(seed=seed)
        return env
    # ... original gym.make() path, completely unchanged below
```

---

### The test model: `assets/robot.xml`

A minimal 3-DOF robot arm is included at `assets/robot.xml` with three hinge joints, a named `end_effector` site, a named `target` site, and a `hazard1` sphere geom.

---

### Running it

```bash
export MUJOCO_GL=egl   # required on WSL2

python -m src.train \
  --env_id mujoco:assets/robot.xml \
  --algo lagppo \
  --total_timesteps 50000 \
  --seed 0 \
  --num_envs 1 \
  --cost_budget 0.05 \
  --lr_lambda 5e-4 \
  --eval_freq 5000 \
  --log_dir logs/mujoco_lagppo
```

---

### Limitations

**`--num_envs 1` is required.**
`MujocoRoboticEnv` cannot be pickled across subprocesses, so `SubprocVecEnv` will crash.

**The shield is a pass-through.**
`MujocoShield` satisfies the `ShieldingActionWrapper` interface but does not do geometry-based action projection. The existing 2D geometric shield is designed for point robots in a flat plane and is not directly applicable to a 3D arm.

**Tested on MuJoCo 2.3.0.**
The `mujoco.viewer` module was added in 2.3.7. On 2.3.0, rendering is skipped silently.

**Site names must match the XML.**
If your model does not have sites named `end_effector` and `target`, the distance-based reward returns zero.

---

## Quickstart
1) **Create an environment and install dependencies**
- Install system MuJoCo (see "Environment setup" below)
- Create and activate a clean Python 3.10+ environment (`conda` or `venv`)
- Install Python dependencies

2) **Run a minimal end-to-end example (100k steps)**

```bash
bash scripts/quickstart.sh
```

This will:
- Train **Lagrangian PPO** with a conservative shield on `SafetyPointPush1-v0` for 100k steps
- Evaluate the resulting checkpoint
- Produce plots under `results/`

## Environment setup

- **Python**: >=3.10
- **MuJoCo**: `mujoco>=2.3` is installed via `pip`, but MuJoCo **system dependencies** must be installed separately. See: [https://github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- **GPU**: Optional; PyTorch will use CUDA if available

## Installation

With **conda**:

```bash
conda create -n .venv python=3.10 -y
conda activate .venv
pip install -r requirements.txt
```

With **venv**:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## System Specifications

Experiments in the paper were run on:

- **CPU:** AMD Ryzen 7 3700X (8 cores, 16 threads)
- **GPU:** RTX 3070 (8GB VRAM)
- **RAM:** 32 GB DDR4
- **OS:** Ubuntu 24.04 on WSL2

## How to reproduce the paper experiments

To reproduce all 24 conditions from the paper (4 algorithms $\times$ 2 shield conditions $\times$ 3 seeds):

```bash
nohup bash run_full_ablation.sh > logs/run_full_ablation.log 2>&1 &
echo "Running as PID $!"
```

with `nohup` it keeps running even if you close the terminal or SSH session. 
All output goes to logs/run_full_ablation.log. You can check progress anytime with:

```bash
tail -f logs/run_full_ablation.log
```

And check which run is currently active with:

```bash
grep "===" logs/run_full_ablation.log
```

To run a single condition (example: LagPPO with shield, seed 0):

```bash
python -m src.train --env_id SafetyPointPush1-v0 \
  --algo lagppo --total_timesteps 1000000 \
  --seed 0 --num_envs 4 --eval_freq 20000 \
  --cost_budget 0.05 --lr_lambda 5e-4 --use_shield \
  --log_dir logs/lagppo_shield_on_seed0 \
  --ckpt_dir checkpoints/lagppo_shield_on/seed_0
```

To evaluate all trained checkpoints:

```bash
for algo in ppo sac rcpo lagppo; do
  for condition in shield_off shield_on; do
    for seed in 0 1 2; do
      python -m src.evaluate \
        --env_id SafetyPointPush1-v0 \
        --model_path checkpoints/${algo}_${condition}/seed_${seed}/latest \
        --episodes 20 --seed 100 \
        --log_dir results/eval_${algo}_${condition}_seed${seed}
    done
  done
done
```

To reproduce all plots:

```bash
python -m src.visualize \
  --log_dirs logs/ppo_shield_off_seed{0,1,2} \
             logs/ppo_shield_on_seed{0,1,2} \
             logs/sac_shield_off_seed{0,1,2} \
             logs/sac_shield_on_seed{0,1,2} \
             logs/rcpo_shield_off_seed{0,1,2} \
             logs/rcpo_shield_on_seed{0,1,2} \
             logs/lagppo_shield_off_seed{0,1,2} \
             logs/lagppo_shield_on_seed{0,1,2} \
  --out_dir results/final_plots
```

## How to run training (general)

For a full list of options:

```bash
python -m src.train --help
python -m src.evaluate --help
```

The most important flags:

- `--env_id`: which Safety-Gymnasium env to use
- `--algo`: `ppo`, `sac`, `lagppo`, `rcpo`
- `--use_shield`: turn the geometric shield on
- `--cost_budget`, `--lr_lambda`: Lagrangian CMDP hyperparameters
- `--penalty_coef`: fixed penalty coefficient for RCPO

Example commands:

```bash
# LagPPO with shield
python -m src.train --env_id SafetyPointPush1-v0 --algo lagppo \
  --total_timesteps 200000 --seed 0 --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --num_envs 4 --eval_freq 10000 --log_dir logs/pointpush_lagppo

# PPO baseline (no constraints)
python -m src.train --env_id SafetyPointPush1-v0 --algo ppo \
  --total_timesteps 200000 --seed 0 --num_envs 4 --eval_freq 10000

# SAC baseline
python -m src.train --env_id SafetyPointPush1-v0 --algo sac \
  --total_timesteps 200000 --seed 0 --num_envs 4 --eval_freq 10000

# RCPO (fixed penalty)
python -m src.train --env_id SafetyPointPush1-v0 --algo rcpo \
  --total_timesteps 200000 --seed 0 --penalty_coef 1.0
```

## Visualization

Learning curves and safety metrics:

```bash
python -m src.visualize --log_dirs logs/pointpush_lagppo logs/pointpush_ppo \
  --out_dir results/plots
```

## Reproducibility checklist
- Deterministic seeding for Python, NumPy, and PyTorch (`src/utils/seeding.py`)
- Package versions recorded in checkpoint config JSON
- Exact command lines and hyperparameters logged to CSV and TensorBoard
- Results saved under:
  * `logs/` (raw scalars)
  * `runs/` (TensorBoard)
  * `results/` (plots, metrics CSV)
  * `checkpoints/` (trained models)
- Three-seed protocol with shaded 95% confidence intervals in plots

## Troubleshooting

- **MuJoCo errors**: ensure system MuJoCo is correctly installed. Try `MUJOCO_GL=egl` or `osmesa` for headless setups
- **Environment not found**: double-check the exact Safety-Gymnasium env ID and package version
- **CUDA / GPU issues**: force CPU with `CUDA_VISIBLE_DEVICES=""` or `--device cpu`

## Future updates

Roughly in order of priority:

- **LagPPO stability**: more systematic sweep over `--cost_budget` and `--lr_lambda`; saved default configs per env
- **Environment coverage**: properly test and document SafetyPointButton1-v0, SafetyCarPush1-v0, FetchPush-v2
- **Training quality of life**: add checkpointing + resume (including Lagrange multiplier state); improve logging
- **Longer-term**: more seeds, more environments, evaluation on additional Safety-Gymnasium tasks

## Notes on other simulators (e.g. Isaac Lab)

Everything in this repo is built around **MuJoCo + Gymnasium / Safety-Gymnasium**.

If you want to experiment with other simulators (e.g. Isaac Lab), the rough steps would be:

- add a new environment factory in `src/envs/make_env.py`
- ensure environments expose rewards and costs matching the current interfaces
- adapt the shield code if moving from 2D point/car robots to full 3D manipulators

This is not implemented yet.

## Background reading

- Altman, *Constrained Markov Decision Processes*, 1999 - CMDP basics
- Tessler et al., *Reward Constrained Policy Optimization*, arXiv:1805.11074 - RCPO
- Ray et al., *Benchmarking Safe Exploration in Deep RL*, 2019 - Safety Gym
- Ames et al., *Control Barrier Functions: Theory and Applications*, ECC 2019 - safety via barrier functions
- Ji et al., *Safety-Gymnasium*, arXiv:2310.12567 - the benchmark used in this work

## Citation

If you use this repository in academic work, please cite the paper (see `CITATION.cff`) or:

```
Pesacane, S. (2026). Geometric Shielding and Lagrangian Constraints are Complementary:
An Empirical Study in Safe Reinforcement Learning.
https://github.com/samuelepesacane/Safe-Reinforcement-Learning-for-Robotic-Manipulation/
```

## License

MIT License (see `LICENSE`).
