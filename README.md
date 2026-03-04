# Safe Reinforcement Learning for Robotic Control

**Note from the author**  
This repo is my personal playground for safe RL in robotics, combining CMDP-style Lagrangian methods with simple geometric shields and (synthetic) preference-based rewards.
It's a research prototype, not a polished library.

## Project status & context

This is a research prototype on a single 8GB-GPU machine.
The focus is:

- studying how Lagrangian CMDPs, fixed penalties (RCPO-style), shields and preference rewards trade off return vs safety
- keeping the code small and readable so it's easy to change and extend

RCPO is currently a simple fixed-penalty baseline, and preferences use synthetic labels. They're there as baselines and extension points, not finished systems.

## Current preliminary results

The plots in `results/plots/` come from a single ablation run on one machine (8GB GPU, relatively short training). 
They're there mainly to check that the pipeline works end-to-end, not as "final" numbers.

A few quick takeaways:

- **LagPPO reacts to safety, but it's not tuned yet.**  
  Returns are reasonable, but average cost and violation rate are still high.
  The dual updates clearly do something, just not in a well-calibrated way.

- **Very tight budgets can make things worse.**  
  With a budget like '0.01', both cost and CVaR blow up. 
  The dual step is too aggressive relative to the target, so $\lambda$ chases the constraint and destabilizes training instead of enforcing it.

- **Loose budgets buy return by spending safety.**  
  Relaxed budgets ('0.1') give better returns but keep costs high. 
  If you tell the algorithm "safety is cheap", it will happily pay for reward.

- **Preference-based rewards are still a toy here.**  
  With synthetic labels and short runs, the learned reward doesn't help yet and sometimes hurts return. 
  This part is mostly a proof of concept.

- **Baselines behave roughly as expected.**  
  PPO: decent return, unsafe.  
  SAC: more conservative but under-trained.  
  RCPO (fixed penalty): quite safe but very conservative in return.

- **Shield stats are not trustworthy yet.**  
  The intervention curve is flat at zero, which is almost certainly an issue
  in how interventions are logged, not a sign that the shield never fires.

So for now, the code path works, the trends make sense, but these runs are closer to smoke tests than to a proper benchmark. 
Longer runs and a bit of tuning are still missing.

## Project Overview

This project investigates **safe reinforcement learning (Safe RL)** algorithms for robotic control tasks, with a focus on balancing reward maximization and constraint satisfaction. Multiple algorithms are implemented and evaluated:

- **Baseline algorithms**: PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic).
- **Constrained algorithms**: LagPPO (Lagrangian-based PPO), RCPO (Reward Constrained Policy Optimization).
- **Preference-based extensions**: optional human-in-the-loop preference learning.
- **Ablation studies**: systematic variation of hyperparameters and components to analyze their effect.

Experiments are run in **Safety-Gymnasium** environments (e.g. `SafetyPointPush1-v0`) where agents must achieve goals while minimizing violations of safety constraints during exploration.

## Motivation

Safe RL is one of the main bottlenecks for real robots: you want the agent to explore and learn, but you don't want it to smash into everything on the way.

This repo is my attempt to glue together a few simple ideas:

- **Lagrangian CMDPs**: treat safety as a cost with a budget and update a Lagrange multiplier online.
- **Geometric shields**: a small 2D keep-out module that projects actions away from known hazards.
- **Preference-based rewards**: learn a reward model from comparisons instead of hand-crafting everything (here I start with synthetic preferences).

The point isn't to squeeze out the best possible score, but to see how these pieces interact under realistic compute and noisy safety signals.

## Repository layout

This is the rough structure of the code:

- `src/train.py`: main training entry point (PPO, SAC, LagPPO, RCPO baseline)
- `src/evaluate.py`: offline evaluation on the *true* environment, writes `metrics.csv`
- `src/ablations.py`: small ablation launcher (wraps train + eval), used by `scripts/train_all.sh`
- `src/envs/make_env.py`: environment factory (Safety-Gymnasium, optional Gymnasium-Robotics) + shield wiring
- `src/algos/lagppo.py`: Lagrangian PPO bits (`LagrangianState`, callback)
- `src/safety/shield.py`: geometric "keep-out" shield for 2D point/car robots
- `src/safety/metrics.py`: helpers to aggregate returns/costs/violation rate/CVaR into tables
- `src/reward/preferences/*`: tiny preference-learning module (data buffer, MLP reward model, wrappers)
- `scripts/quickstart.sh`: one small LagPPO + shield run on `SafetyPointPush1-v0`
- `scripts/train_all.sh`: runs the ablations and plots summary figures under `results/`

If you're trying to hack on something:
- **LagPPO / $\lambda$ updates** start from `src/algos/lagppo.py`
- **Shield** `src/safety/shield.py` and `src/envs/make_env.py`
- **Preferences** `src/reward/preferences/`

## Supported Environments

This project uses [**Safety-Gymnasium**](https://github.com/PKU-Alignment/Safety-Gymnasium), a benchmark library for **safe reinforcement learning (Safe RL)**.  
Unlike standard RL benchmarks that only optimize reward, Safety-Gymnasium introduces **safety costs** (e.g. entering hazard zones) and **constraint thresholds** that the agent should satisfy while learning.  
Full documentation: [Safety-Gymnasium Docs](https://safety-gymnasium.readthedocs.io/en/latest/introduction/about_safety_gymnasium.html)

Main environments I'm using:
- **SafetyPointPush1-v0**: A point robot pushing a box to a goal while avoiding hazards  
- **SafetyPointButton1-v0**: A point robot navigating to a button with hazards in the way  
- **SafetyCarPush1-v0**: A car-like robot that has to push a box while avoiding hazards  
- Optional (no safety costs): **gymnasium-robotics FetchPush-v2**. This environment has not been tested yet. You can find the YAML file under 'configs/fetch_push.yaml' (see FetchPush-v2 below)

These environments give:
- **Reward** for doing the task (reach goal, push box, press button)  
- **Cost** for unsafe events (stepping into hazard regions)  
- **Budget** that algorithms like LagPPO or RCPO attempt to respect

## Gymnasium-Robotics Environments

In addition to Safety-Gymnasium, this project includes optional support for [**Gymnasium-Robotics**](https://gymnasium.farama.org/environments/robotics/), a set of continuous-control benchmarks for robotic manipulation tasks.

Unlike Safety-Gymnasium, **Gymnasium-Robotics environments do not include explicit safety costs**. They are designed primarily for **task performance** (reaching, pushing, or placing objects). This makes them useful for testing baseline performance and comparing pure reward optimization against Safe RL setups.

### FetchPush-v2

- **Task**: A simulated **Fetch robotic arm** (7-DOF arm with parallel gripper) pushes a small box across a table to a target goal location  
- **Observation space**: robot joint positions and velocities, gripper state, object positions  
- **Action space**: 4D continuous control (three end-effector velocity commands + gripper open/close)  
- **Reward function**: sparse or dense, depending on configuration. Typically:
  - '0' if the box is at the goal position (success).
  - '-1' at each timestep otherwise (sparse).
- **Goal specification**: typically used with **Hindsight Experience Replay (HER)** for improved sample efficiency in sparse-reward settings.  

**Why include FetchPush-v2?**

- It provides a **baseline manipulation task without safety costs**, allowing direct comparisons between standard RL training and safety-constrained setups  
- It is useful for testing whether the Safe RL infrastructure can generalize to non-safety environments  

> **Note:**
> FetchPush-v2 support is **optional** and has **not been fully tested** in this repository yet.
> A configuration file is provided at `configs/fetch_push.yaml`.

---

## Direct MuJoCo XML Integration

This section talks about a direct use of MuJoCo. You don't need this for Safe RL. 
This is a section if you want to go over Safety-Gymnasium and use your personal robot.
If you want you can skip this section and go directly to [Quickstart](#Quickstart).  

---

Up to this point, every environment in this repo goes through **Safety-Gymnasium**.

That is the right choice for studying safe RL algorithms: Safety-Gymnasium gives you reward,
cost, hazard geometry, and a well-defined constraint budget out of the box.
You don't have to think about any of that, you just call `gym.make("SafetyPointPush1-v0")`
and get a fully wired environment.

But there is a cost to that convenience.
You are always working with someone else's robot, someone else's reward function,
and someone else's definition of what "safe" means.

If you want to ask:
- *What if I have a different robot model?*
- *What if I want to define my own hazard zones from scratch?*
- *What if I want to test the pipeline on a model I designed or received from somewhere?*

then you need to go one level below Safety-Gymnasium and talk to **MuJoCo directly**.

That is what `mujoco_connector.py` does.

---

### What MuJoCo actually is (and what Safety-Gymnasium hides from you)

MuJoCo is a physics engine.
You describe a robot and a world in an XML file, and MuJoCo simulates the physics:
joint positions, velocities, contact forces, everything.

When you call `gym.make("SafetyPointPush1-v0")`, Safety-Gymnasium:
1. loads a MuJoCo `.xml` model internally (you never see it)
2. builds observations by flattening joint states for you
3. applies a reward function it already wrote
4. applies a cost function it already wrote (hazard violations, etc.)
5. wraps all of that in a standard Gymnasium interface

So the Gymnasium interface you interact with in this codebase is already several layers above raw MuJoCo.

The connector cuts those layers out.
It loads any `.xml` file you give it, talks to `mujoco.MjModel` and `mujoco.MjData` directly,
and builds a Gymnasium environment from scratch that the rest of this pipeline can use without modification.

---

### What the connector does

`mujoco_connector.py` defines one main class: `MujocoRoboticEnv`.

It is a standard `gymnasium.Env` that:

- loads any MuJoCo `.xml` model by path
- builds the **observation** as: joint positions ∥ joint velocities ∥ end-effector position ∥ goal position
- builds the **action** as: normalized torques in `[-1, 1]`, then rescaled to each actuator's physical range (`ctrlrange` from the XML)
- computes **reward** as: negative distance from end-effector to goal, plus a success bonus when the goal is reached
- computes **cost** as: penetration depth into user-defined hazard geometries, clipped to `[0, 1]`
- returns an **info dict** with exactly the keys the rest of the pipeline expects:
  `cost`, `is_success`, `shield_intervened`

That last point is important.
`CostInfoWrapper`, `LagrangianCallback`, `ShieldingActionWrapper`, `evaluate.py`, `metrics.py` 
none of these know or care whether the environment came from Safety-Gymnasium or from a raw XML file.
They just see a Gymnasium env that produces observations, rewards, costs, and info dicts in the right format.
The connector produces exactly that.

---

### How the integration works

The natural place to add a new environment type in this codebase is `src/envs/make_env.py`.

Looking at the actual code, `make_env()` is mostly a wrapper assembler.
The real environment creation happens inside `_try_make()`, which calls `gym.make(env_id)`.

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

If the `env_id` starts with `mujoco:`, we load the XML directly and return early.
If not, we fall through to the original path.

The second edit is three lines in `build_shield_factory()` inside `src/train.py`.
That function normally reads `env.unwrapped.world.hazards_pos`, a Safety-Gymnasium internal attribute
that does not exist on a raw MuJoCo env. The guard skips that and returns a pass-through shield instead:

```python
def build_shield_factory(env_id: str):
    if env_id.startswith("mujoco:"):
        from mujoco_connector import MujocoShield
        return lambda env: MujocoShield()
    # ... original factory code, unchanged
```

That is the complete integration. Two files edited, five lines each.
Every algorithm, every wrapper, every evaluation script above that level is untouched.

---

### The test model: `assets/robot.xml`

A minimal 3-DOF robot arm is included at `assets/robot.xml`.

It has:
- three hinge joints (shoulder rotation, elbow, wrist)
- a named `end_effector` site (what the connector tracks as the tip of the arm)
- a named `target` site (the goal the arm has to reach)
- a `hazard1` sphere geom (the unsafe zone that produces a cost signal)

This model is intentionally simple.
The goal is not to benchmark a production robot but to verify that the full pipeline,
training, evaluation, logging, Lagrangian constraint tracking, runs end-to-end on a model you control.

The site names (`end_effector`, `target`) are the connector's defaults.
If you use a different model, you can override them in `MujocoRoboticEnv.__init__`.
If a site is not found, the corresponding component of reward or observation returns zero
(the simulation still runs, it just has no spatial objective).

---

### Running it

```bash
export MUJOCO_GL=egl   # required on WSL2, avoids display errors

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

Evaluate the trained checkpoint:

```bash
python -m src.evaluate \
  --env_id "mujoco:assets/robot.xml" \
  --model_path "checkpoints/mujoco:assets/robot.xml/lagppo/seed_0/latest" \
  --episodes 10 --seed 100 \
  --log_dir results/eval_mujoco
```

---

### What the results actually show (and what they mean)

Running LagPPO for 50k steps on the included arm model:

| metric                               | value |
|--------------------------------------|---|
| policy std (start $\to$ end)         | 1.0 $\to$ 0.68 |
| value loss (start $\to$ end)         | 2.96 $\to$ 0.34 |
| explained variance (start $\to$ end) | −0.24 $\to$ +0.21 |
| evaluation success rate              | 1.0 |
| average cost                         | 0.0 |
| Lagrange multiplier $\lambda$        | 0.0 (flat) |

The policy clearly learns: the std shrinks (the agent becomes more decisive),
the value loss drops (the critic learns to predict returns),
and the explained variance goes from negative to positive
(which means the critic's predictions are becoming meaningful).

The cost and lambda being flat at zero deserves a short explanation,
because at first glance it might look like something is broken.

It is not.

The hazard geom (`hazard1`) is placed at a position the arm does not need to pass through
to reach the target. The policy learns a path that reaches the goal without entering the hazard zone.
When no constraint is ever violated, the Lagrangian has nothing to react to, so $\lambda$ stays at zero.
That is correct behaviour.

There is a useful concept here worth naming: a constraint that is never violated is called
**vacuously satisfied**. It does not mean the safety mechanism is not working,
it means the task geometry made the constraint inactive.
If you move `hazard1` into the arm's natural reaching path in the XML,
the cost signal becomes non-zero and the Lagrangian mechanism will activate.
The cost curve will no longer be flat.

---

### Limitations

**`--num_envs 1` is required.**  
`MujocoRoboticEnv` cannot be pickled across subprocesses, so `SubprocVecEnv`
(which this codebase uses when `num_envs > 1`) will crash.
This makes training slower than the Safety-Gymnasium runs.

**The shield is a pass-through.**  
`MujocoShield` satisfies the `ShieldingActionWrapper` interface but does not do
any geometry-based action projection. The existing 2D geometric shield in `src/safety/shield.py`
is designed for point robots in a flat plane, it is not directly applicable to a 3D arm.
Extending it to 3D manipulation is a non-trivial research problem in its own right.

**Tested on MuJoCo 2.3.0.**  
The `mujoco.viewer` module (used for live rendering) was added in 2.3.7.
On 2.3.0, rendering is skipped silently. Everything else works.

**Site names must match the XML.**  
If your model does not have sites named `end_effector` and `target`,
the distance-based reward returns zero. The simulation runs normally,
it just has no spatial objective until you configure the right site names.

---

### What this is actually for

This connector is not trying to replace Safety-Gymnasium for safe RL benchmarking.
Safety-Gymnasium is the right tool for that and this repo uses it for everything else.

What the connector demonstrates is that the pipeline is not locked to one simulator.

The same Lagrangian PPO, the same cost tracking, the same evaluation metrics,
the same Lagrange multiplier update, all of it runs on a model you define from scratch in XML.
The algorithms do not know what kind of environment they are training in.
They just see a Gymnasium interface.

That is the useful property for research:
if you want to study safe RL on a new robot geometry,
you write the XML, point `--env_id` at it, and everything else stays the same.

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
- **MuJoCo**: `mujoco>=2.3` is installed via `pip`, but MuJoCo **system dependencies** must be installed separately depending on your OS. See:
  [https://github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- **GPU**: Optional; PyTorch will use CUDA if available (Good luck without GPU)
- **Other simulators (e.g. Isaac Lab)**: not used here. This repo currently targets MuJoCo + Gymnasium only.

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

## System Specifications and Computational Constraints

Experiments in this repository were run on:

- **CPU:** AMD Ryzen 7 3700X (8 cores, 16 threads)
- **GPU:** RTX 3070 (8GB VRAM)
- **RAM:** 32 GB DDR4
- **OS:** Ubuntu 24.04 on WSL2 (Windows Subsystem for Linux) 

Safe RL training is **computationally intensive**, especially for ablation sweeps across multiple algorithms, seeds, and hyperparameters.

## How to run training
For a full list of options:

```bash
python -m src.train --help
python -m src.evaluate --help
python -m src.ablations --help
```

The most important flags:

- `--env_id`: which Safety-Gymnasium env to use
- `--algo`: `ppo`, `sac`, `lagppo`, `rcpo`
- `--use_shield`: turn the geometric shield on
- `--cost_budget`, `--lr_lambda`: Lagrangian CMDP hyperparams
- `--use_preferences`, `--pref_steps`: enable the synthetic preference reward model


Lagrangian PPO on `SafetyPointPush1-v0` with shield and cost budget:

```bash
python -m src.train --env_id SafetyPointPush1-v0 --algo lagppo \
  --total_timesteps 200000 --seed 0 --cost_budget 0.05 --lr_lambda 5e-4 \
  --use_shield --num_envs 4 --eval_freq 10000 --log_dir logs/pointpush_lagppo
```

Baseline PPO (no constraints):

```bash
python -m src.train --env_id SafetyPointPush1-v0 --algo ppo \
  --total_timesteps 200000 --seed 0 --num_envs 4 --eval_freq 10000
```

SAC baseline:

```bash
python -m src.train --env_id SafetyPointPush1-v0 --algo sac \
  --total_timesteps 200000 --seed 0 --num_envs 4 --eval_freq 10000
```

RCPO (fixed penalty):

```bash
python -m src.train --env_id SafetyPointPush1-v0 --algo rcpo \
  --total_timesteps 200000 --seed 0 --penalty_coef 10.0
```

Preference-based reward learning (synthetic preferences):

```bash
python -m src.train --env_id SafetyPointPush1-v0 --algo lagppo \
  --use_preferences --pref_steps 20000 --total_timesteps 200000 --seed 0
```

**Evaluation:**

```bash
python -m src.evaluate --env_id SafetyPointPush1-v0 \
  --model_path checkpoints/SafetyPointPush1-v0/lagppo/seed_0/latest.zip \
  --episodes 20 --seed 100 --log_dir results/eval_pointpush
```


## Visualization

Learning curves and safety metrics:

```bash
python -m src.visualize --log_dirs logs/pointpush_lagppo logs/pointpush_ppo \
  --out_dir results/plots
```

Ablation summary (after running ablations):

```bash
python -m src.visualize --ablations_csv results/ablations_summary.csv \
  --out_dir results/plots
```

## Ablations

To run a full ablation sweep:

```bash
SEEDS="0 1 2 3 4" TIMESTEPS=1000000 OUT_CSV="results/test.csv" PARALLEL=5 \
  bash scripts/train_all.sh
```

Parameters:

* `SEEDS`: list of random seeds (affects reproducibility)
* `TIMESTEPS`: number of training timesteps per run
* `OUT_CSV`: output CSV file summarizing results
* `PARALLEL`: number of jobs to run concurrently (`--parallel` in `ablations.py`)


The ablations include:
1) Shield on vs off (with LagPPO)
2) Lambda learning rate: {1e-4, 5e-4, 1e-3}
3) Cost budget: {0.01, 0.05, 0.1} per step
4) Preference reward vs environment reward (LagPPO)
5) Algorithm comparison: PPO, SAC, RCPO, LagPPO

### Resource Constraints

The full configuration above can saturate GPU/CPU memory on a single machine (this happened both on WSL2 and on a native Ubuntu 24.04 setup).

More stable configurations used in practice:

- **Single-seed, longer run** - reduces robustness (no averaging across seeds):

  ```bash
  SEEDS="0" TIMESTEPS=500000 PARALLEL=1
  ```

- **Multi-seed, shorter runs (compromise)**:

  ```bash
  SEEDS="0 1 2" TIMESTEPS=300000 PARALLEL=1
  ```

Trade-offs of the compromise configuration:

- Fewer gradient updates per run (shorter training)
- Fewer seeds for averaging
- Higher variance and potentially less stable performance than the ideal setting

Current limitations:

- Single-machine training with **8GB VRAM**
- No checkpoint resume: interrupted runs restart from scratch
- Reduced total timesteps compared to the original plan (`5 seeds x 1M steps`)

## Reproducibility checklist
- Deterministic seeding for Python, NumPy, and PyTorch (`src/utils/seeding.py`)
- Package versions recorded in checkpoint config JSON
- Exact command lines and hyperparameters logged to CSV and TensorBoard
- Results saved under:
  * `logs/` (raw scalars)
  * `runs/` (TensorBoard)
  * `results/` (plots, metrics CSV)
  * `checkpoints/` (trained models)
- Synthetic preferences used by default for preference-learning experiments
- Three-seed protocol recommended for plots (with shaded 95% confidence intervals)

## Troubleshooting

- **MuJoCo errors**: ensure system MuJoCo is correctly installed and visible. Try `MUJOCO_GL=egl` or `osmesa` for headless setups
- **Environment not found**: double-check the exact Safety-Gymnasium env ID and package version
- **CUDA / GPU issues**: if you hit CUDA version mismatches, you can force CPU with `CUDA_VISIBLE_DEVICES=""` or pass `--device cpu`

## Future updates

This repo is still very much a work in progress. Roughly in order of pain points:

- **LagPPO stability and defaults**
  - Do a more systematic sweep over `--cost_budget` and `--lr_lambda` (and clipping) instead of guessing.
  - Save a couple of "good enough" default configs per env (YAML) so people don't have to rediscover them.

- **Shield logging and evaluation**
  - Fix the intervention logging bug so every modified action is actually counted (right now plots lie).
  - Add a few targeted tests (boundary cases, multiple hazards) and one small experiment comparing:
    - no shield
    - shield only
    - LagPPO only
    - LagPPO + shield

- **Preference pipeline cleanup**
  - Make the synthetic preference code less hacky (segment length, sampling, etc. as flags).
  - Add a small script/notebook to check how the learned reward correlates with env reward/cost.
  - Longer term: plug in real human preferences instead of only synthetic labels.

- **Environment coverage and configs**
  - Properly test and document which environments are supported (Point/Button/Car + FetchPush-v2).
  - Add small config files per env (seeds, horizon, budgets, shield settings) so runs are reproducible.

- **Training / tooling quality of life**
  - Add checkpointing + resume (including Lagrange multiplier state).
  - Improve logging: make it easier to see rewards, costs, $\lambda$, and shield interventions in one place.
  - Add a few more unit tests (Lagrangian update, shield projection, preference loss) and maybe a tiny Continuous Integration (CI) job.

### Longer-term ideas (if/when compute allows)

- Memory optimization for large-batch or long-horizon runs
- Distributed training over multiple GPUs or nodes
- Extended evaluation on additional Safety-Gymnasium tasks and robotics simulators

## Notes on other simulators (e.g. Isaac Lab)

Right now everything in this repo is built around **MuJoCo + Gymnasium / Safety-Gymnasium**.

If you want to experiment with other simulators (e.g. Isaac Lab), the rough steps would be:

- add a new environment factory in `src/envs/make_env.py` that creates vectorized Isaac Lab environments
- make sure they expose rewards, costs, and any safety signals in a way that matches the current interfaces
- adapt the shield code if you move from 2D point/car robots to full 3D manipulators

This is not implemented yet. At the moment, consider it a "future experiment" rather than an advertised feature.

## Background reading (if you're new to Safe RL)

If some of the terms above are unfamiliar (CMDP, RCPO, CVaR, preference learning), a few classic references:

- Altman, *Constrained Markov Decision Processes*, 1999 - CMDP basics
- Tessler et al., *Reward Constrained Policy Optimization*, ICLR 2019 - RCPO-style methods
- Ray et al., *Benchmarking Safe Exploration in Deep RL*, NeurIPS 2019 - Safety Gym
- Christiano et al., *Deep RL from Human Preferences*, NeurIPS 2017 - preference-based reward learning
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems*, TAC 2019 - safety via barrier functions

The repo is not trying to reproduce their results, but it borrows a lot of ideas from that literature.

Also, I greatly encourage you to explore [**Safety-Gymnasium**](https://github.com/PKU-Alignment/Safety-Gymnasium).

## Citation

If you use this repository in academic work, please cite:

See `CITATION.cff`.

## License

MIT License (see `LICENSE`).
