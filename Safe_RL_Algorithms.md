# Safe Reinforcement Learning and Algorithms Used in This Project

This file explains the theory behind the algorithms used in this repository.

The goal is **not** to present a fully formal treatment of reinforcement learning.
Instead, the objective is to provide enough intuition to understand how the training pipeline works and how the different safety mechanisms interact.

The explanations focus on:

* how the algorithms behave during training
* how reward and safety signals are used
* how the geometric shield interacts with the learning process

The algorithms used in this project are:

* **PPO**: baseline policy gradient method
* **SAC**: off-policy actor-critic baseline
* **RCPO**: fixed-penalty safe RL baseline
* **Lagrangian PPO (LagPPO)**: the main constrained algorithm

All experiments are run in **Safety-Gymnasium environments**, where agents must complete a task while minimizing safety violations.

These environments provide both:

* **reward signals** for task progress
* **safety cost signals** for constraint violations

which makes them suitable for studying safe reinforcement learning.

---

# 1. Safe Reinforcement Learning

In standard reinforcement learning the objective is simple:

maximize the expected return.

$$
J_R(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}[G_0]
$$

where the return $G_0$ is the discounted sum of rewards collected during an episode.

In this setup the agent only cares about **task performance**.

If unsafe behavior leads to higher reward, the algorithm will happily learn it.

For real robotic systems this is usually unacceptable.

A robot exploring its environment may:

* collide with objects
* enter restricted zones
* apply unsafe forces
* damage itself or the environment

Safe reinforcement learning addresses this by adding **explicit safety constraints** to the learning problem.

---

# 2. Constrained Markov Decision Processes

Safe RL is commonly formulated as a **Constrained Markov Decision Process (CMDP)**.

A CMDP extends the usual MDP by introducing a **cost signal** in addition to reward.

At each timestep the environment produces:

* reward $r_t$: measures task performance
* cost $c_t$: measures safety violations

The objective becomes:

maximize reward while keeping cost below a specified budget.

Formally we write

maximize

$$$$
J_R(\theta) = \mathbb{E}[G_0^R]
$$

subject to

$$
J_C(\theta) = \mathbb{E}[G_0^C] \le d
$$

where

* $G_0^R$ is the reward return
* $G_0^C$ is the cost return
* $d$ is the allowed safety budget

Intuitively:

* reward encourages task completion
* cost penalizes unsafe behavior
* the constraint limits how much safety violation is acceptable.

---

# 3. How Safety-Gymnasium Provides Costs

In the environments used in this project the cost signal is produced automatically by the simulator.

For example in **SafetyPointPush1-v0**:

* the agent receives reward for pushing a box to a goal
* the environment contains **hazard regions**
* entering a hazard produces a cost

So during training each step returns

```
obs, reward, terminated, truncated, info
```

where

```
info["cost"]
```

contains the safety cost.

Algorithms designed for safe RL track these costs during training and attempt to keep the **expected cost below a specified budget**.

---

# 4. Algorithms Used in This Project

This project evaluates four algorithms.

Two of them are **standard RL algorithms**:

* PPO
* SAC

Two are **safe RL algorithms**:

* RCPO
* Lagrangian PPO

The standard algorithms act as baselines.

The constrained algorithms attempt to enforce safety constraints during learning.

---

# 5. Proximal Policy Optimization (PPO)

PPO is a **policy gradient algorithm** designed for stable training of neural network policies.

It belongs to the **actor-critic family**:

* the **policy (actor)** selects actions
* the **value function (critic)** estimates expected returns

---

## 5.1 Basic idea

Policy gradient methods directly optimize the policy parameters $\theta$.

The gradient of the objective has the form

$$
\nabla_\theta J(\theta) =
\mathbb{E}\left[
\nabla_\theta \log \pi_\theta(a_t|s_t) A_t
\right]
$$

where $A_t$ is the **advantage estimate**.

The intuition is simple:

* increase probability of actions that produced high advantage
* decrease probability of actions that produced low advantage

However naive policy gradient updates can be unstable.

A single update step may change the policy too much.

PPO solves this by **limiting how far the policy can move in one update**.

---

## 5.2 PPO clipped objective

PPO introduces a probability ratio

$$
r_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

which measures how much the new policy differs from the old one.

The objective becomes

$$
L^{clip} =
\mathbb{E}
\left[
\min(
r_t A_t,
\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t
)
\right]
$$

The clipping prevents the ratio from growing too large.

This keeps policy updates **conservative and stable**.

---

## 5.3 PPO training loop

A typical PPO iteration looks like this:

1. collect trajectories with the current policy
2. compute advantage estimates
3. update the policy using the PPO objective
4. update the value network

Because PPO uses trajectories generated by the current policy, it is an **on-policy algorithm**.

---

## 5.4 PPO in this project

In this project PPO serves as a **baseline without safety constraints**.

The policy is trained only to maximize reward.

As a result:

* it often learns high-return behaviors
* but it frequently violates safety constraints

This provides a useful comparison point when evaluating safe RL algorithms.

---

# 6. Soft Actor-Critic (SAC)

SAC is an **off-policy actor-critic algorithm** designed for continuous control.

---

## 6.1 Maximum entropy RL

SAC optimizes a modified objective:

maximize reward while also maximizing policy entropy.

$$
J =
\mathbb{E}
\left[
\sum r_t + \alpha H(\pi(\cdot|s_t))
\right]
$$

The entropy term encourages exploration and prevents the policy from becoming deterministic too early.

---

## 6.2 Actor-critic structure

SAC maintains:

* a stochastic policy network
* two Q-value networks

The Q-networks estimate expected returns

$$
Q(s,a)
$$

The policy learns to choose actions that maximize the Q-values while maintaining entropy.

---

## 6.3 Replay buffer

Unlike PPO, SAC is **off-policy**.

This means transitions are stored in a replay buffer:

$$
(s,a,r,s')
$$

Training samples random minibatches from this buffer.

Benefits include:

* higher sample efficiency
* more stable gradient updates.

---

## 6.4 SAC in this project

SAC acts as a **continuous control baseline**.

It can reach high reward but does not explicitly enforce safety constraints.

---

# 7. Reward Constrained Policy Optimization (RCPO)

RCPO is one of the simplest safe RL approaches.

The constraint is converted into a **penalty in the reward function**.

$$
r'_t = r_t - \lambda c_t
$$

where:

* $c_t$ is the cost
* $\lambda$ is a penalty coefficient

The algorithm then runs a normal RL algorithm using the modified reward.

---

## Example

Suppose:

```
reward for goal = +10
hazard cost = 1
$\lambda$ = 5
```

Then

```
r' = 10 − 5 × 1 = 5
```

Unsafe trajectories become less attractive.

However the penalty coefficient is **fixed**, so safety is controlled manually.

---

# 8. Lagrangian PPO (LagPPO)

LagPPO solves the CMDP using a **Lagrangian relaxation**.

Instead of fixing a penalty, the algorithm introduces a **Lagrange multiplier** $\lambda$.

The objective becomes

$$
L(\theta,\lambda) =
J_R(\theta) - \lambda (J_C(\theta) - d)
$$

where $d$ is the cost budget.

---

## 8.1 Multiplier intuition

The multiplier acts like a **price for safety violations**.

If cost exceeds the budget:

* $\lambda$ increases
* unsafe actions become more expensive

If cost is below the budget:

* $\lambda$ decreases
* the algorithm focuses more on reward.

---

## 8.2 Combined advantage

LagPPO modifies the PPO update using a combined advantage

$$
A_t = A_t^{reward} - \lambda A_t^{cost}
$$

This pushes the policy to:

* increase reward
* reduce cost.

---

## 8.3 Dual update

The multiplier is updated after each batch

$$
\lambda \leftarrow \lambda + \eta (J_C - d)
$$

This allows the algorithm to **adapt the safety penalty automatically**.

---

# 9. Interaction with the Geometric Shield

The project optionally uses a **geometric action shield**.

The shield sits **between the policy and the environment**.

If an action would move the robot into a hazard region, the shield modifies it before execution.

Safe actions pass through unchanged.

Unsafe actions are projected onto a safe direction.

The shield operates entirely at **execution time**.

---

## What the shield knows

The shield assumes:

* hazards are circular regions
* hazard centers are known
* hazard radii are known

Using this information it can predict whether an action would enter a hazard.

---

## Action correction

If an action would enter a hazard:

* the component pointing toward the hazard is removed
* the remaining motion stays tangential or outward

This produces a corrected action that stays outside the unsafe region.

---

## What the policy experiences

The policy **does not know the shield exists**.

From the algorithm's perspective:

* the policy outputs an action
* the environment returns the resulting state, reward, and cost.

If the shield intervenes, the transition reflects the corrected action.

---

# 10. How the Training Pipeline Works

A training step in this project looks like:

```
state → policy → action → shield → environment
                         ↓
              next_state, reward, cost
```

Then:

1. the transition is stored or processed
2. the algorithm updates the policy

Different algorithms use the reward and cost signals differently.

---

# 11. Common Failure Modes in Safe RL Training

Safe RL introduces additional challenges.

Some common issues include:

### Lagrange multiplier explosion

If the cost is consistently above the budget, $\lambda$ can grow rapidly.

Large $\lambda$ values can destabilize training.

---

### Extremely tight budgets

If the budget is too small, the task may become impossible.

The algorithm may learn overly conservative behavior.

---

### Loose budgets

If the budget is too large, the constraint becomes inactive.

The algorithm behaves like standard RL.

---

### Sparse cost signals

If violations are rare, the algorithm receives little safety feedback.

Learning the constraint becomes slow.

---

### Shields hiding violations

If the shield always corrects unsafe actions, the algorithm may underestimate how unsafe the policy actually is.

---

### Vacuous constraints

Sometimes the task can be solved without ever triggering hazards.

In this case the constraint is satisfied automatically and does not influence training.

---

# 12. Why Safe RL Is Difficult in Robotics

Safe RL becomes especially challenging when applied to real robotic systems.

Several factors contribute to this difficulty.

---

## Exploration vs safety

Learning requires exploration.

However exploration can lead to dangerous states.

A robot must learn new behaviors while avoiding harmful actions.

---

## Sparse safety feedback

Many safety violations are rare events.

This makes it difficult for the algorithm to learn reliable safety signals.

---

## Physical constraints

Robots have limited torque, speed, and precision.

Small modeling errors can produce unexpected behavior.

---

## Sensor noise

Robots rely on sensors that may be noisy or delayed.

This uncertainty makes safety constraints harder to enforce.

---

## Sim-to-real transfer

Policies are often trained in simulation.

However the real world behaves differently.

Small discrepancies can produce safety violations after deployment.

---

## Multiple safety layers

Real robots often include multiple safety systems:

* motion planners
* safety controllers
* learned policies

Understanding how these layers interact is a major research challenge.

---

# 13. Summary

This project studies safe reinforcement learning by combining:

* standard RL baselines (PPO, SAC)
* constrained algorithms (RCPO, LagPPO)
* a geometric runtime shield

The goal is to understand how these mechanisms interact during training.

In particular the experiments investigate the trade-offs between:

* task performance
* safety constraint satisfaction
* runtime safety interventions.
