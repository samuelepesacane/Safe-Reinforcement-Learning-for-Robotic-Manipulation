# Safe Reinforcement Learning for Robotic Manipulation (Draft)

**Author:** Samuele Pesacane

---

## Abstract

In this project I built an open-source framework for safe reinforcement learning in robotic control / robotic manipulation-style settings, mainly on top of Safety-Gymnasium. The goal was not to invent a new algorithm, but to put the main pieces in one place and actually test how they behave together under realistic constraints (limited compute, short runs, noisy safety signals).

The framework includes PPO, SAC, a simple RCPO-style fixed-penalty baseline, and Lagrangian PPO (LagPPO) in a unified training/evaluation pipeline. On top of that, I added a conservative geometric action shield (to push actions away from predicted hazard incursions) and a small preference-based reward learning module (synthetic preferences for now, mainly as a prototype).

The main result from my ablations is not "everything is solved" (it is not). It is more like: the theory is nice, but in practice safety-performance tradeoffs show up very quickly. In particular, across my short-horizon runs, SAC was often more conservative than PPO and LagPPO (lower average cost / violation rate), and changing the cost budget / dual learning rate had a very strong effect on whether LagPPO behaved sensibly or just oscillated. This is exactly the kind of thing I wanted to study: how safety guarantees get weaker once function approximation, finite samples, and imperfect environment models enter the picture.

---

## 1. Introduction

Safe RL is one of the main bottlenecks if we want RL to be useful for real robots.

In simulation, the agent can explore badly and nobody cares. On hardware, "exploration" can mean collisions, unsafe motions, wear on components, or just behavior that is too unreliable to trust. Standard RL only cares about maximizing return. It does not care whether the agent reached the goal safely or by doing something reckless ten times before it got lucky.

This repo is my attempt to build a small but usable safe RL playground where I can study that tradeoff directly.

The idea was to combine a few simple ingredients:

1. **Constrained RL (CMDP style)** with a Lagrangian multiplier  
   (so the agent gets reward, but also pays for unsafe behavior)

2. **A geometric keep-out shield**  
   (so obviously unsafe actions can be corrected before they hit the environment)

3. **Preference-based reward learning**  
   (so I can later replace hand-designed reward with something learned from comparisons)

I built everything around **Safety-Gymnasium**, because it already gives me the right setup: task reward + safety cost (hazard incursions), which is perfect for constrained RL experiments.

This is still a draft-stage project. The code path works end-to-end, but the point right now is not leaderboard performance. The point is to understand what breaks, what helps, and what is too sensitive to trust without more tuning.

---

## 2. What I implemented

### 2.1 Unified training/evaluation framework

I wanted one codebase where I could swap algorithms and safety components without rewriting everything each time.

So the framework has:

- **PPO** (baseline)
- **SAC** (baseline)
- **RCPO-style baseline** (currently a fixed cost penalty, not a full RCPO implementation)
- **Lagrangian PPO (LagPPO)**

and one shared pipeline for:

- training
- checkpointing
- logging
- evaluation
- ablation runs
- plotting

This sounds simple, but this was actually a big part of the work: making wrappers, logging, and evaluation consistent enough that I can compare methods without constantly patching the scripts.

---

### 2.2 Lagrangian PPO (LagPPO)

The constrained RL part follows the standard CMDP / Lagrangian idea.

The agent maximizes reward while trying to respect a cost budget. In practice, I shape the reward as:

$$
r'_t = r_t - \lambda c_t
$$

where:

- $r_t$ = task reward
- $c_t$ = safety cost (from Safety-Gymnasium, e.g. entering hazard zones)
- $\lambda$ = Lagrange multiplier (updated online)

The dual update is:

$$
\lambda \leftarrow \max(0,\ \lambda + \alpha(\hat{c} - d))
$$

where:

- $\hat{c}$ is the empirical **average per-step cost** over a rollout window
- $d$ is the target per-step cost budget
- $\alpha$ is the dual learning rate

So the logic is:

- if the policy is too unsafe, $\lambda$ goes up (cost hurts more)
- if the policy stays below budget, $\lambda$ can go down

In code, this is implemented with a small `LagrangianState` helper + a callback that reads costs from `info["cost"]`, updates $\lambda$, and logs both cost and multiplier over time.

One thing I saw very clearly: **LagPPO is sensitive** to the budget and to the dual LR. If the budget is too tight or $\alpha$ is too aggressive, the multiplier can overreact and training becomes unstable instead of safer.

---

### 2.3 Geometric action shield (keep-out shield)

Lagrangian methods work in expectation, which is useful, but they do not stop a single obviously bad action.

So I added a simple geometric shield for 2D Safety-Gym tasks:

- it stores hazard regions as circles
- reads the agent's 2D position from the observation
- treats the action as a local velocity command
- predicts the next position
- if the predicted next position enters a hazard, it rescales / projects the action so the agent stays just outside the hazard boundary

This is intentionally simple. It is not a formal safety guarantee and it does not model long-horizon dynamics. It is basically a "don't step straight into the hazard" filter.

I like this design because it is cheap and easy to reason about. It is also a nice extension point if I want to test something more advanced later (barrier-function style shields, learned forward models, etc.).

One issue I still need to clean up: **intervention logging is not fully reliable yet** in some wrapper combinations, so the shield stats in the plots are not always trustworthy.

---

### 2.4 Preference-based reward learning (prototype)

I also added a small preference-learning module because I wanted the framework to support "reward is learned" experiments, not only hand-designed rewards.

Right now it works like this:

1. collect random trajectories
2. split them into short segments
3. sample segment pairs
4. create **synthetic preferences** (the segment with higher env return is preferred)
5. train a small MLP reward model with a Bradley-Terry style loss
6. replace the environment reward with the learned reward during RL training

At this stage, this is mostly a **plumbing prototype**, not a polished human-feedback pipeline. Since the labels are synthetic and runs are short, the learned reward is often just a noisy approximation of the environment reward.

Still, I wanted this in the repo now because later I can swap the synthetic labels with real preferences without changing the whole training loop.

---

## 3. Environments and setup

The main experiments use **Safety-Gymnasium**, especially:

- `SafetyPointPush1-v0`
- `SafetyPointButton1-v0`
- `SafetyCarPush1-v0` (optional / secondary)

These are good for this project because they provide:

- task reward
- safety cost
- hazard geometry (which also makes shielding easy to prototype)

I also included optional support for **Gymnasium-Robotics FetchPush-v2**, but that environment does not provide explicit safety costs, so it is more of a future extension for manipulation experiments than a core benchmark in this draft.

### Why I still call this "robotic manipulation"
Even though most of the current results are from SafetyPoint / SafetyCar tasks, the framework was designed with robotic manipulation in mind:

- unified safe RL algorithms
- reward replacement hooks
- shield wrapper interface
- evaluation with safety metrics

So the current experiments are more like a safe RL benchmark layer, and later I can reuse the same structure on more realistic manipulation environments.

---

## 4. Evaluation and metrics

I did not want to evaluate only on return, because that hides the whole point of safe RL.

So the framework computes safety-aware metrics such as:

- **average return**
- **average episodic cost**
- **violation rate** (episodes with non-zero cost)
- **per-step cost**
- **CVaR cost** (tail-risk cost, useful for bad episodes)
- **success rate** (when available)
- **shield interventions** (when logging is consistent)

A small thing I care about here: evaluation is done on the **raw environment** (not with reward shaping active), because I want to judge the learned policy behavior, not the training trick used to get it.

---

## 5. Ablation study design

I set up a systematic ablation driver with five dimensions:

1. **Shield on/off**
2. **Cost budget** (tight / medium / loose)
3. **$\lambda$ learning rate** (dual LR)
4. **Algorithm choice** (PPO, SAC, RCPO baseline, LagPPO)
5. **Preference-based rewards on/off**

This is maybe the part I'm happiest with in the codebase, because once the pipeline works, I can run controlled comparisons instead of changing ten things at once and guessing what happened.

I also kept the scripts pretty simple (`src/ablations.py`, `scripts/train_all.sh`) so I can rerun everything later with longer horizons and more seeds.

---

## 6. Main observations (from current runs)

This is the important part: what did I actually learn from the current experiments?

### 6.1 Theoretical safe RL ideas degrade fast under short runs / limited compute

This was the biggest takeaway.

On paper, constrained RL gives a clean story: optimize reward under a cost budget. In practice, with function approximation + noisy rollouts + short training horizons, the behavior is much messier.

The framework made this visible very clearly:

- constraints are often not satisfied under tight budgets
- dual updates can become unstable
- safety metrics can move in the wrong direction if hyperparameters are not calibrated

This is exactly why I think naive deployment on hardware is risky. If a method only looks safe in idealized training settings, that is not enough.

---

### 6.2 SAC was often more conservative than PPO and LagPPO in my runs

One result that came out pretty consistently in my ablation summaries is that **SAC tended to achieve lower average cost and lower violation rates than PPO and LagPPO**, even when returns were not always the best.

I do not read this as "SAC is universally better for safe RL."  
I read it as:

- under my current compute limits
- with short training horizons
- and with the current tuning

SAC behaved more conservatively than the PPO-based variants.

That was useful to see, because it reminded me not to assume that a constrained method (like LagPPO) will automatically dominate a strong baseline if the constrained method is under-tuned.

---

### 6.3 Cost budget and dual LR matter a lot (and not always in the way theory suggests)

I swept cost budgets and dual learning rates, and the results made the tradeoff very obvious.

A few patterns I observed:

- **Very tight budgets** can make LagPPO unstable  
  (the multiplier chases the constraint too aggressively)

- **Relaxing the budget** often increased task return  
  (less penalty pressure, easier optimization)

- In my ablation summary, relaxing the budget from very tight to looser values also **reduced average episodic cost** in some runs, which is counterintuitive but real in this setup  
  (my interpretation: the very tight setting destabilized learning so much that the agent behaved worse overall, including safety)

So the "budget" is not just a high-level safety preference. It is also a training-stability knob.

---

### 6.4 Shielding is promising, but I need cleaner intervention logging

Qualitatively, the shield helps, especially early in training. It prevents some of the dumbest hazard incursions and makes exploration less chaotic.

But quantitatively, I am not fully happy with the current shield stats yet because the intervention counter can be flat in some runs when it clearly should not be.

So for now my position is:

- **the shield is useful**
- **the code path works**
- **the logging needs one more cleanup pass** before I trust the intervention plots fully

---

### 6.5 Preference-based rewards are a good extension point, not a strong result yet

The preference module is working, but with synthetic labels and short runs it is still a toy.

Sometimes it behaves fine, sometimes it hurts return a bit, and right now it does not give me a strong safety improvement.

That is okay for this draft. I mainly wanted:

- a reward model
- segment pairing
- a training loop
- an environment wrapper for reward replacement

Now that those pieces exist, I can later plug in real human preferences (or better synthetic protocols) without rewriting the project.

---

## 7. What this framework is useful for

Even though the results are still preliminary, the framework already does something I wanted:

It lets me systematically study where the clean theory of safe RL starts to break when I include:

- function approximation
- limited samples
- short training runs
- imperfect shielding
- model-environment mismatch

That is the real contribution of this project for me right now.

It is not a polished benchmark paper yet.  
It is a working research scaffold that exposes the safety-performance tradeoff in a way I can inspect and improve.

---

## 8. Limitations

A few things are still clearly incomplete:

- **Shield is 2D and local**  
  Good for SafetyPoint / SafetyCar style tasks, not enough for full 3D manipulation safety.

- **RCPO path is a baseline**  
  It is currently a fixed penalty baseline, not a full adaptive RCPO implementation.

- **Preference labels are synthetic**  
  Useful for testing the pipeline, but not the same as human feedback.

- **Compute limits matter**  
  Many runs are short by design (single machine, practical time limits), so the results are more "stress test" than final benchmark.

- **No real hardware yet**  
  This is all simulation, so the final sim-to-real gap is still open.

---

## 9. Conclusion

I developed an open-source framework for safe reinforcement learning in robotic control / manipulation-style environments, with:

- PPO, SAC, RCPO-style baseline, and Lagrangian PPO in one pipeline
- a geometric keep-out action shield
- a preference-based reward learning module
- safety-aware evaluation and ablation tooling

The main thing I learned is that safety methods that look clean in theory can degrade a lot in practice under realistic compute limits and short horizons. In my runs, this showed up as strong sensitivity to the cost budget and dual LR, and as baseline methods (especially SAC) sometimes looking safer than expected compared to constrained PPO variants.

That is not a failure of the project. It is exactly the result I wanted to expose.

Now I have a framework where I can study these effects systematically, instead of just assuming the constraints will work because the math says they should.

---

## 10. Next steps

What I want to do next:

1. Fix shield intervention logging properly
2. Run longer ablations (more timesteps, more seeds)
3. Tune LagPPO more carefully (budget + dual LR)
4. Add a stronger RCPO implementation (not only fixed penalty)
5. Replace synthetic preferences with a small real preference collection loop
6. Push the same framework into a more realistic manipulation environment

That should give me a much stronger next version (with cleaner tables/plots and less "prototype" caveats), while keeping the same overall structure.
