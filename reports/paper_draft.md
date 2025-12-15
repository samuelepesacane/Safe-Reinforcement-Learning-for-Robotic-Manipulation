# Constrained Exploration for Robotic Manipulation with Lagrangian PPO and Action Shielding

**Author:** Samuele Pesacane

---

## Abstract

This is a small Safe RL project on MuJoCo-style control tasks using Safety-Gymnasium.  
I combine a constrained MDP approach (Lagrangian PPO with an average cost budget), a simple geometric action shield that tries to keep the agent out of hazard regions, and a tiny preference-based reward model trained from synthetic segment comparisons.

The code includes PPO, SAC, a fixed-penalty RCPO-style baseline, and Lagrangian PPO (LagPPO), plus some safety-aware metrics and ablation scripts.  
On the SafetyPoint tasks I tried so far, LagPPO plus shielding tends to reduce constraint violations and cost CVaR while keeping returns in a reasonable range, although the results are noisy and definitely not "state of the art". This draft is mostly a record of what I implemented and what I observed, not a final paper.

---

## 1. Introduction

Reinforcement learning can do very well in simulation, but getting a learned policy to behave sensibly on a real robot is another story. Exploration can break hardware, collide with things, or just behave in ways you wouldn't want around humans. Standard RL optimizes expected return and doesn't care *how* the agent gets that reward, as long as the numbers go up.

Constrained Markov decision processes (CMDPs) offer one standard way to handle this: you get both a reward and a cost, and you try to keep the cost under some budget. In theory this gives you "optimal and safe" behavior; in practice, especially with function approximation and limited compute, constraints are often violated during training and sometimes even at convergence.

In this project I played with a small Safe RL stack for robotic-style control, built around three pieces:

1. **Lagrangian PPO (LagPPO):** PPO with a Lagrange multiplier on the cost and dual updates that try to enforce a cost budget.
2. **Geometric action shielding:** a 2D "keep-out" projection that tries to prevent the agent from stepping into hazard discs.
3. **Preference-based reward learning:** a simple reward model trained from pairwise segment comparisons (synthetic for now).

Everything runs on Safety-Gymnasium (e.g. `SafetyPointPush1-v0`) with Stable-Baselines3 (PPO, SAC) as backbones.  
The goal is not a new Safe RL algorithm, but rather to see how these ingredients interact under realistic constraints (single machine, limited GPU, noisy safety signals) and to have a codebase I can iterate on.

---

## 2. Background

### 2.1 Constrained Markov Decision Processes

A CMDP augments the usual MDP with a cost and a constraint:

- State space \( \mathcal{S} \), action space \( \mathcal{A} \), transition kernel \( P \), reward \( r(s,a) \).
- A **cost** function \( c(s,a) \geq 0 \) measuring safety violations.
- The agent maximizes expected discounted return \( \mathbb{E}[\sum_t \gamma^t r_t] \) subject to
  \[
  \mathbb{E}\left[\sum_t \gamma^t c_t\right] \leq d,
  \]
  where \( d \) is a cost budget.

A common trick is to form the Lagrangian
\[
\mathcal{L}(\pi, \lambda) = \mathbb{E}\left[ \sum_t \gamma^t \big(r_t - \lambda c_t\big) \right] + \lambda d,
\]
and alternate between improving the policy under a **shaped reward** \( r' = r - \lambda c \) and updating the dual variable \( \lambda \) based on how much the constraint is violated.

In code, this becomes "run PPO on \( r' \)" plus "every so often, look at the average cost and nudge \(\lambda\) up or down".

### 2.2 Lagrangian Methods for Safe RL

In practice I use PPO as the base algorithm and update the Lagrange multiplier using an empirical average cost:

\[
\lambda \leftarrow \max\left( 0, \lambda + \alpha \big(\hat{c} - d\big) \right),
\]

where \( \hat{c} \) is the average per-step cost over the last rollout, \( d \) is the target per-step budget, and \( \alpha \) is a dual learning rate.

- If the agent is too unsafe (\( \hat{c} > d \)), \(\lambda\) increases and cost gets penalized more strongly.
- If the agent is below budget, \(\lambda\) tends to decrease towards zero.

In the current implementation, I found that the method is **quite sensitive** to both \( \alpha \) and the choice of budget \( d \). Too aggressive \( \alpha \) or too tight a budget often makes \(\lambda\) "chase" the constraint and destabilize training.

### 2.3 Action Shielding

Lagrangian methods work on **expected** cost. They do not directly stop individual unsafe actions from being executed. To reduce the most obvious failures, I added a simple **geometric action shield** for 2D robots:

- The shield stores circular hazard regions in the workspace.
- From the observation it extracts the robot's 2D position.
- For a proposed action (treated as a 2D velocity), it predicts the next position:
  \[
  x_{\text{next}} = x + \Delta t \, a_{xy}.
  \]
- If \( x_{\text{next}} \) lands inside a hazard disc, the shield rescales the action along its direction so that the next position lies just outside the hazard boundary.

This is very local and approximate: it doesn't model real dynamics or long-term safety, but it can cut down on "drive straight into the hazard" behavior early in training.

> **TODO:** right now, shield intervention logging is buggy in some runs (often flat zero), so the plots underestimate how often the shield actually fires.

### 2.4 Preference-Based Reward Learning

Reward design is another source of problems. For this project I implemented a basic **preference-based reward learning** module, mostly as a hook for future human feedback:

1. Collect trajectories from a random policy.
2. Sample pairs of short segments (e.g. 25 steps each).
3. Assign a synthetic label: the segment with higher cumulative environment reward is "preferred".
4. Train a small MLP that maps observations to a scalar reward, using a Bradley-Terry loss so that preferred segments get higher predicted cumulative reward.
5. Wrap the environment so that training uses the learned reward instead of the original one.

Right now the labels are fully synthetic, so this is more about testing the plumbing than about real human alignment. In experiments with limited data and short training, the learned reward can be quite noisy and doesn't always help.

---

## 3. Method

### 3.1 Lagrangian PPO (LagPPO)

LagPPO in this repo is implemented by combining Stable-Baselines3 PPO with a small `LagrangianState` helper:

- The environment exposes a scalar cost in `info["cost"]`.
- Over each rollout of length \( T \), we compute
  \[
  \hat{c} = \frac{1}{T} \sum_{t=1}^T c_t.
  \]
- The Lagrange multiplier is updated via
  \[
  \lambda \leftarrow \max\big(0, \lambda + \alpha (\hat{c} - d)\big),
  \]
  with simple clipping to avoid numerical explosion.

The shaped reward passed to PPO is
\[
r'_t = r_t - \lambda c_t.
\]

`LagrangianState` stores \(\lambda\), the learning rate \(\alpha\), the budget \(d\), and an update cadence (roughly the rollout length). A callback periodically:

- reads average cost from recent transitions,
- updates \(\lambda\),
- logs both the updated multiplier and the observed costs.

> **Note:** in some runs, very tight budgets plus high \(\alpha\) lead to oscillatory or unstable behavior. I haven't tuned this thoroughly yet.

### 3.2 Geometric Keepout Shield

The `GenericKeepoutShield` sits next to the environment and modifies actions before they hit the simulator:

- It keeps a list of hazard discs \((x_i, y_i, r_i)\).
- It reads a 2D position from the observation (e.g. `"agent_pos"`).
- For action \( a \), it uses the first two components \( a_{xy} \) and predicts:
  \[
  x_{\text{next}} = x + \Delta t \, a_{xy}.
  \]
- If this point is inside a hazard disc, it scales the action by a factor \( t \in [0, 1] \) found via a small 1D search so that the next position sits just outside the disc.

This is meant to be a cheap, "don't step into the lava" style mechanism, not a safety guarantee. Shields are attached via `make_env`, and the environment is expected to record an intervention flag so we can count how often the shield acts.

> **TODO:** intervention logging is not fully consistent across all wrappers yet; this is visible in ablation plots where interventions ~= 0 even when the shield should be doing something.

### 3.3 Preference-Based Reward Model

The preference-learning pipeline follows a standard pattern, but at a small scale:

1. **Random rollouts:** collect trajectories of `(obs, action, reward)` from a random policy.
2. **Segments:** break trajectories into fixed-length segments (e.g. 25 steps).
3. **Pairs + labels:** sample pairs of segments and label the one with higher cumulative reward as preferred.
4. **Reward network:** train a small MLP on flattened observations with a Bradley-Terry style loss so that preferred segments get higher predicted cumulative reward.
5. **Wrapper:** during training, use a `RewardReplacementWrapper` to replace environment reward with the learned reward at each step.

In the current experiments, this mostly serves as a "toy" preference model. With the limited number of segments and short runs, it doesn't reliably improve safety or performance, but it shows how a proper human-in-the-loop module could plug in later.

### 3.4 Overall System and Implementation

The main training script is `src/train.py`:

- Driven by CLI flags: `--algo {ppo,sac,lagppo,rcpo}`, `--env_id`, `--use_shield`, `--cost_budget`, `--lr_lambda`, `--use_preferences`, etc.
- Uses vectorized environments (Dummy/SubprocVecEnv) with a shared Lagrange multiplier for LagPPO across workers.
- RCPO is currently implemented as a fixed-penalty wrapper:
  \[
  r'_t = r_t - \beta c_t
  \]
  with a constant \(\beta\) from the CLI.
- Logging is done via a custom JSON/TensorBoard logger.

Evaluation is in `src/evaluate.py`, which:

- loads the SB3 checkpoint (PPO or SAC),
- runs deterministic rollouts on the *raw* environment (no shaping, no shield),
- aggregates safety-aware metrics into a small `metrics.csv`.

---

## 4. Experimental Setup

### 4.1 Environments

Main environments from Safety-Gymnasium:

- **SafetyPointPush1-v0:** point robot pushes a box to a target while avoiding hazards.
- **SafetyPointButton1-v0:** point robot must reach and press a button with hazards in between.
- **SafetyCarPush1-v0 (optional):** car-like robot with differential drive dynamics and hazard regions.

There is also a config for **FetchPush-v2** (Gymnasium-Robotics), which has no cost signals. It's mainly there as a possible extension for "pure task" manipulation experiments; I haven't really used it in the current Safe RL runs.

### 4.2 Algorithms and Baselines

The code supports four variants:

- **PPO:** standard baseline, no cost.
- **SAC:** off-policy baseline with entropy regularization.
- **RCPO baseline:** PPO with fixed penalty \( r' = r - \beta c \).
- **LagPPO:** PPO with Lagrangian cost shaping and dual updates on \(\lambda\).

All share the same basic MLP architecture and mostly default SB3 hyperparameters. The idea is to keep the backbone fixed and change only how costs are handled.

### 4.3 Training Protocol

Typical settings (subject to change as I iterate):

- Training steps: usually in the \( 1\text{e}5\)-\(3\text{e}5\) range per run.
- Up to 4 parallel envs.
- 3-5 random seeds for ablations under ideal conditions; fewer seeds for quick tests.
- Hardware: single PC, RTX 3070 (8GB), WSL2 + Ubuntu.

There's a `scripts/quickstart.sh` script that runs a small LagPPO + shield experiment on `SafetyPointPush1-v0`, evaluates it, and makes some plots. This is mostly a smoke test.

> **Note:** long runs with many seeds can easily hit memory or time limits on this setup, so some ablations are intentionally small.

---

## 5. Metrics

Per episode, I record:

- **Return:** sum of environment rewards.
- **Cost:** sum of safety costs.
- **Violation rate:** how often cost is non-zero (per step or per episode).
- **Length:** episode length in steps.
- **Shield interventions:** how many actions were modified by the shield.
- **Success:** whether the task goal was achieved (when available).

The module `src/safety/metrics.py` turns these into:

- averages and standard deviations,
- CVaR-style tail costs (e.g. mean of top 10% highest-cost episodes),
- simple tables for plotting and comparison.

All metrics are computed on the **unshaped environment**, i.e., no extra penalties and, for now, no shield during evaluation. This is important: shaping and shielding are training-time tools, not how we judge final behavior.

---

## 6. Results (Qualitative Summary)

The exact numbers depend on the run, but, across multiple experiments, a few patterns keep showing up:

- **LagPPO vs PPO:**  
  LagPPO usually lowers cost and violation rate compared to plain PPO, especially with moderate budgets. Returns are sometimes slightly worse but not catastrophic. When \(\alpha\) or the budget are badly chosen, things can go sideways (see the tight-budget ablations).

- **RCPO baseline:**  
  With a small penalty, RCPO behaves almost like PPO (unsafe). With a large penalty, it becomes very conservative and sacrifices most return. There is no obvious "right" \(\beta\), and I did not do a full sweep. LagPPO's adaptive \(\lambda\) feels nicer conceptually, but still requires careful tuning.

- **Shielding:**  
  Qualitatively, the shield helps prevent "drive straight through the hazard" behavior, especially early. Quantitatively, the current intervention logging is not reliable enough to claim strong results here, so I treat this as a promising, but not fully measured, piece.

- **Preferences:**  
  With synthetic labels and short runs, the learned reward roughly recovers the environment reward, but does not clearly improve anything. In some ablations it even hurts return a bit. For now, this module is more of a prototype for future human-feedback experiments than a useful component on its own.

---

## 7. Ablation Studies

The ablation driver (`src/ablations.py` + `scripts/train_all.sh`) explores:

1. **Shield on/off (LagPPO):** how much the geometric shield changes early violations and final safety metrics.
2. **Dual learning rate \(\alpha\):** values in roughly \(\{1\text{e}{-4}, 5\text{e}{-4}, 1\text{e}{-3}\}\), to see how quickly constraints are enforced vs how unstable training becomes.
3. **Cost budget \(d\):** tight (0.01), moderate (0.05), loose (0.1) budgets.
4. **Preference vs environment reward:** same setup, but training on a learned reward instead of the original ones.
5. **Algorithm comparison:** PPO, SAC, RCPO baseline, LagPPO.

The script launches each training run, then calls the evaluation script, and finally merges each `metrics.csv` into a single summary CSV. Plots are generated from that summary via `src/visualize.py`.

> **TODO:** The current ablation plots already show sensible qualitative trends, but they come from relatively short runs and few seeds. A more "paper-like" version would need more compute and time.

---

## 8. Limitations and Future Work

Some obvious limitations of the current state:

- **Shield approximation:**  
  The shield uses very simple 2D geometry and known hazard locations. No dynamics, no 3D, no contact modeling. It's fine for SafetyPoint-style tasks, but not a general safety mechanism.

- **Synthetic preferences only:**  
  Using environment reward to generate preferences is convenient but a bit circular. Real human feedback could reveal different trade-offs (comfort, risk aversion, etc.), but that's not implemented yet.

- **Compute and tuning:**  
  Runs are relatively short and use limited seeds. I did not do large-scale hyperparameter searches, so many "bad behaviours" might be fixable with more careful tuning that I haven't had the time/compute to do.

- **Simulation only:**  
  Everything happens in simulation. No real-robot tests, no sim-to-real study, and no hardware-specific issues.

Future directions I'd like to try (time and compute permitting):

- More robust dual-update schemes and better heuristics for picking the budget.
- Extending shields to 3D tasks (possibly with barrier functions or learned models).
- Plugging in real human preferences instead of synthetic labels.
- Running longer, multi-seed experiments to get cleaner statistics.

---

## 9. Conclusion

This project implements a small Safe RL stack for robotic control, combining:

- Lagrangian PPO (LagPPO),
- a simple geometric action shield, and
- an optional preference-based reward model,

all built on top of Safety-Gymnasium and Stable-Baselines3.

The main takeaway so far is that dual-based CMDP methods and simple shields can be made to work together, but they are sensitive to budgets, learning rates, and logging details. The current results are closer to a "sanity-check stage" than to a final benchmark, but the code path works end-to-end and is easy to extend.

My hope is that this repo can serve as a starting point, for myself and possibly others, to iterate on safer, learning-based control methods, rather than as a finished product.

---

## References

- E. Altman. *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999.  
- A. Ray et al. "Benchmarking Safe Exploration in Deep Reinforcement Learning." *Safe RL Workshop at NeurIPS*, 2019.  
- C. Tessler et al. "Reward Constrained Policy Optimization." *ICLR*, 2019.  
- P. Christiano et al. "Deep Reinforcement Learning from Human Preferences." *NeurIPS*, 2017.  
- A. D. Ames et al. "Control Barrier Function Based Quadratic Programs for Safety Critical Systems." *IEEE Transactions on Automatic Control*, 2019.
