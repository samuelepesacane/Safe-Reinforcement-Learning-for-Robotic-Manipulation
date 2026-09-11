# Safe RL — Code Audit and Submission Execution Plan

**Written:** 2026-08-07
**Author of analysis:** session working from `SAFE_RL_PROJECT_REPORT.md`, the preprint PDF, and a
direct read of the source and logs in `rl_project_riemannian-shield-extended/`.
**Purpose:** freeze the findings of the pre-submission code audit and the agreed execution plan so
they never have to be re-derived. This document supersedes `SAFE_RL_PROJECT_REPORT.md` §11 and the
README's "key findings" section wherever they conflict — **this document was verified against the
code and the logs; those were not.**

---

## 0. Provenance and verification status

Everything below was checked against source files and `logs/*/metrics.jsonl`, not from memory or
from the write-ups. Line numbers refer to `rl_project_riemannian-shield-extended/`.

**Verified directly:**
- Observation/action space dimensions — decoded from the pickled `observation_space` inside the
  saved SB3 checkpoints (`checkpoints/*/latest.zip` → `data` → base64 → `pickletools.dis`).
- λ trajectories, cost trajectories, intervention rates — parsed from all `logs/*/metrics.jsonl`.
- Eval metrics — aggregated from all `results/eval_*/metrics.csv`.
- Per-run wall time — from `stat` mtimes on the sequential ablation logs.
- Branch equivalence — `diff -rq` on `src/`; `diff -q` on sampled log files.

**NOT verified (no Python env available on the audit machine):**
`safety_gymnasium`, `stable_baselines3`, and `torch` are **not installed** on the Windows box where
this audit ran. Therefore:
- The exact Safety-Gymnasium accessor names for agent position and heading
  (`env.unwrapped.task.agent.pos` / `.mat` / body quaternion) are **unconfirmed**. Resolving these
  is the first implementation step (see §5, Run 0a probe).
- No code was executed against a live environment.

Branch note: `rl_project_branch_main/` and `rl_project_riemannian-shield-extended/` have
**byte-identical `logs/`, `checkpoints/`, and `results/`**. `src/` differs only in `train.py`,
`callbacks/eval_callback.py`, and the three files new to the extension
(`safety/riemannian_shield.py`, `safety/__init__.py`, `plot_three_way.py`). The report's §0 claim
that main is "single-environment" is true of its *code*, not of its *data directories*.

---

## 1. Six findings, in order of consequence

### D1 — The shield has never seen the agent's position

`src/safety/shield.py:121-123`:

```python
# Flat array: SafetyPointPush1-v0 puts agent XY at indices 0:2
if isinstance(obs, np.ndarray) and obs.shape[0] >= 2:
    return np.array(obs[:2], dtype=np.float32)
```

Actual observation spaces, decoded from the checkpoints:

| env | obs dim | action dim |
|---|---|---|
| SafetyPointPush1-v0 | 76 | 2 |
| SafetyPointGoal1-v0 | 60 | 2 |
| SafetyCarGoal1-v0 | 72 | 2 |

These are standard Safety-Gymnasium layouts: proprioceptive sensors (accelerometer 3 + velocimeter 3
+ gyro 3 + magnetometer 3 = 12; 24 for Car with its extra wheel/ball sensors) plus 16-bin
pseudo-lidar per object class (Push 4×16=64 → 76; Goal 3×16=48 → 60 and 72).

**None of them contain absolute XY position.** `obs[:2]` returns the first two accelerometer
components. In every reported shield-on run the shield computed
`pos_next = accelerometer[:2] + 0.1·a_xy` and tested that against hazard discs.

Supporting evidence: on `goal`, LagPPO shield-on intervenes at rate 0.009 while RCPO shield-on
intervenes at 0.0007 — a 13× spread for identical geometry and identical shield, consistent with
"position" being a policy-dependent sensor reading rather than a position.

`src/tests/test_shield.py` passes because it feeds a **dict** obs `{"agent_pos": ...}`, a layout
none of these environments produce. Green unit test, broken production path.

Root cause of the mistaken belief: preprint §4.1 asserts "The observation space is a flat vector
encoding the agent's position and velocity, the box and goal positions, and pseudo-lidar readings."
That is false for Safety-Gymnasium.

### D2 — Hazard positions are frozen at episode 0

`set_hazards` has exactly one call site: `src/train.py:260`, inside `factory(env)` in
`build_shield_factory`. `src/envs/make_env.py:403` calls `factory` **once**, at env construction.
`ShieldingActionWrapper.reset` (`make_env.py:314-319`) calls `shield.on_reset()`, which only zeroes
intervention counters (`shield.py:81-92`) — it does not re-read hazards.

Safety-Gymnasium re-randomizes the hazard layout on every reset. At 1M steps / 1000-step episodes,
roughly 999 of every 1000 episodes ran against the episode-0 layout.

Preprint §4.3 and Appendix A.3 both state hazards are read "at the start of each episode." They are
not.

### D3 — The Riemannian shield deflects *toward* hazards, and saturates

**Sign.** `src/safety/riemannian_shield.py:147`:

```python
grad_i = (2.0 / (clearance ** 3 * (d + 1e-8))) * diff      # diff = pos - h
```

`diff` points **away** from the hazard, so `grad_i` is the repulsive direction (i.e. −∇φ). Then
line 198:

```python
a_deflected[:2] = a[:2] - deflection      # deflection = alpha * clip(grad, ±max_action_norm)
```

Subtracting the repulsive direction pushes the action **toward** the hazard. One sign flip.

The docstrings contradict each other on this point: line 112 gives the correct
`∇φ_i = -2/(d-r)³ · (pos-h)/d`; line 115 then calls the coded quantity "steepest ascent"; line 146
says "we negate to get the repulsive direction." The code implements the repulsive vector and then
subtracts it.

**Saturation.** `|grad_i| = 2/clearance³`. With `influence_radius = 0.4` (used in every reported
experiment), any contributing hazard has `clearance ≤ 0.4`, so `|grad| ≥ 2/0.064 = 31.25` — always
far above `max_action_norm = 1.0`. The element-wise `np.clip` therefore saturates both components
essentially always, giving `deflection ≈ 0.1 · sign(diff)` component-wise: a fixed-magnitude,
45°-quantized nudge. Not the smooth proximity-proportional deflection the README and the report's §7
describe.

### D4 — The preprint's shield equation is not the implemented shield

Preprint Appendix A.3, Eq. (5):

```
a_safe = a - ((a · (h - x)) / ||h - x||²) (h - x)
```

That is a tangential projection: remove the hazard-ward component, keep the rest.

`src/safety/shield.py:191-196` does something else — bisect a scalar `t ∈ [0,1]` and return
`a_xy * scale`. Isotropic shrink, direction preserved. §3.3 and §4.3 repeat the "keeping only the
tangential or outward component" language.

Different operator, different geometry, different intervention statistics. Neither operator produces
"the closest safe action to the original proposal" as §4.3 claims.

### D5 — λ is inert in every run, by roughly two orders of magnitude

**The maximum λ ever reached in any of the 27 LagPPO runs, any environment, any shield condition,
is 0.0075** (`lagppo_shield_off_seed0`, i.e. Push).

Scale argument on the car: episodic return ≈ 28.8 over 1000 steps → 0.029 reward/step; cost rate
≈ 0.064. For the penalty `λ·c` to be comparable to the reward signal you need **λ ≈ 0.45**. Observed
λ ≈ 0.004 — about **100× too small to change the policy**.

The magnitude is arithmetically forced by the hyperparameters:
`Δλ = lr_lambda × err = 5e-4 × 0.014 ≈ 7e-6` per dual update, × 492 updates ≈ 0.0034. Exactly what
the logs show.

Eval confirms the mechanism did nothing (3 seeds, mean ± std):

| condition | return | per-step cost |
|---|---|---|
| goal_lagppo_shield_off | 25.98 ± 1.12 | 0.046 ± 0.007 |
| goal_ppo_shield_off | 26.06 ± 0.26 | 0.046 ± 0.002 |
| car_lagppo_shield_off | 28.79 ± 1.31 | 0.064 ± 0.002 |
| car_ppo_shield_off | 31.05 ± 0.17 | 0.057 ± 0.003 |

On `goal`, LagPPO and PPO are the same run. On `car`, unconstrained PPO is *safer* than LagPPO.

**Consequence for the anomaly.** This gives the "boring" explanation a specific, well-supported
form: **the dual loop is open.** λ is too small to affect the policy → the policy never reduces cost
→ the error stays positive → λ rises forever. This holds on *all three* environments, is
geometry-free, and currently explains the car anomaly better than the actuation-non-integrability or
representation stories.

**Also undercuts the infeasibility mechanism.** Report §11 claims an achievable cost floor of ~0.051
above the 0.05 budget. 0.051 is the Riemannian *eval* per-step cost, not a floor.
`car_sac_shield_off` reaches 0.052 (shield-on: 0.049) with return 37.2 — so 0.05 sits roughly *at*
the achievable frontier on the car, not below it. Infeasibility is not established.

### D6 — "point robots plateau, car climbs" is not what the logs show

Applying the pre-registered primary statistic (final-20% window, `slope · window_len / mean(λ)`) to
all 27 LagPPO runs:

| run family | S1 normalized slope (3 seeds) | S3 CV resid | S4 window cost | S4 frac windows > budget |
|---|---|---|---|---|
| car lagppo shield_off | 0.180, 0.138, 0.191 | 0.010–0.014 | 0.062–0.066 | 0.64–0.70 |
| car lagppo shield_on | 0.244, 0.186, 0.237 | 0.009–0.014 | 0.068–0.075 | 0.77–0.81 |
| car riemannian | 0.251, 0.102, 0.116 | 0.008–0.011 | 0.061–0.070 | 0.60–0.69 |
| goal lagppo shield_off | **−0.130**, 0.108, 0.079 | 0.015–0.031 | 0.049–0.054 | 0.44–0.48 |
| goal lagppo shield_on | 0.149, 0.089, 0.169 | 0.013–0.051 | 0.052–0.055 | 0.51–0.59 |
| goal riemannian | 0.170, 0.139, 0.189 | 0.012–0.044 | 0.054–0.058 | 0.57–0.65 |
| push lagppo shield_off | 0.141, 0.099, −0.012 | 0.014–0.041 | 0.051–0.072 | 0.31–0.53 |
| push lagppo shield_on | 0.165, 0.013, 0.232 | 0.020–0.043 | 0.055–0.060 | 0.36–0.43 |
| push riemannian | −0.108, 0.654, 0.327 | 0.034–0.582 | 0.045–0.085 | 0.25–0.48 |

Car spans 0.10–0.25; PointGoal spans −0.13–0.19. **They overlap almost completely.** No point-robot
run plateaus — λ is still rising at 1M in nearly every seed. Preprint §5.3's "monotonically
increasing λ" is also wrong in the literal sense: the step-to-step monotone fraction is 0.36–0.49 on
push LagPPO, 0.52–0.63 on goal, 0.64–0.75 on car.

**One statistic does separate car from PointGoal perfectly, 9 vs 9, zero overlap** — the
deceleration ratio S2 = (λ gain in 2nd half) / (λ gain in 1st half):

- **car**, all 9 LagPPO runs: 0.69, 0.78, 0.92, 1.08, 0.82, 0.89, 1.00, 0.69, 0.69 → **all ≥ 0.68**
  (linear to accelerating)
- **goal**, all 9: 0.42, 0.21, 0.29, 0.62, 0.30, 0.12, 0.36, 0.20, 0.25 → **all ≤ 0.63**
  (clearly decelerating)
- **push**, all 9: −0.06, −0.89, 1.20, 0.60, 1.25, 2.58, 0.44, 0.55, 3.36 → no consistent behaviour

**The defensible version of the claim:** on PointGoal λ decelerates and is heading toward a plateau
it does not reach within 1M steps; on CarGoal it does not decelerate at all. Weaker than the current
text, but real and cleanly measurable.

### D7 — `train/ep_cost` is identically zero in every run

`src/callbacks/train_logging_callback.py:71-74`:

```python
ep_info = info.get("episode", None)
if ep_info is not None:
    self._ep_returns.append(float(ep_info.get("r", 0.0)))
    self._ep_costs.append(float(ep_info.get("cost", 0.0)))
```

SB3's `Monitor` writes `info["episode"] = {"r": ..., "l": ..., "t": ...}`. There is **no `"cost"`
key** in that dict — `CostInfoWrapper` writes `info["episodic_cost"]` at the top level
(`make_env.py:145`), not inside `info["episode"]`. So `ep_info.get("cost", 0.0)` returns 0.0 every
time and `train/ep_cost` is 0.0 in all 542-line logs.

Not load-bearing for anything in the preprint (which uses eval cost and `train/avg_cost_per_step`),
but it removes the only in-log route to episode length and should be fixed before the new runs.

**Fix (3 lines, do it as part of Run 0):** read `info["episodic_cost"]` for the cost, and also log
`train/ep_len` from `ep_info["l"]`, which SB3 *does* provide and which is currently discarded.
Without `ep_len` in the logs, S5 has to be reconstructed from `results/eval_*/metrics.csv`.

### D15 — Evaluation never applies the shield: every shield_on eval number in this project measures an unshielded rollout

(Numbering continues from D14, found 2026-09-04 and recorded in `SESSION_HANDOFF.md` §7.5 — figure
defects, not yet moved into this doc. D15 is a new session-4 finding, 2026-09-11.)

`src/evaluate.py:116`:

```python
env = make_env(args.env_id, seed=args.seed)
```

No `use_shield` or `shield_factory` argument is passed, so `make_env`'s own gate at
`make_env.py:402` (`if use_shield and shield_factory is not None:`) never constructs
`ShieldingActionWrapper`. This is **documented as deliberate** in the module docstring
(`evaluate.py:8-10`): "No shield is applied during evaluation by design: the goal is to measure the
true safety behavior of the learned policy, not the behavior of the policy plus a runtime correction
layer." That design choice answers a real question (what did the policy learn) but leaves a different
one — what happens when the trained policy is actually deployed behind the shield, which is the
condition every "shield_on" row in every results table in this project is implicitly claimed to
represent — completely unmeasured.

**Consequence:** every eval row for every shield_on condition, across every run this project has ever
reported — including the extended branch's three-way comparison table in `README.md` — is the
converged policy stepped **without** the shield. `avg_interventions` is 0.0 in literally every eval
row on record, shielded configurations included, because the wrapper that would set
`info["shield_intervened"]` is never constructed at eval time. Shielded deployment, as a system, has
never actually been measured by this codebase; only shielded *training* has.

**This also resolves what looked like an anomaly in session 4's analysis, not a new one:** the car
Riemannian runs showed `train/avg_cost_per_step` climbing to 0.65–0.68 late in training while the
corresponding eval `per_step_cost` stayed at 0.052–0.079. These are not the same policy behaving
differently in two measurements of the same system — **they are two different systems**: the training
number is the shield-plus-policy rollout (where the trap described under the mechanism-mix analysis in
`SESSION_HANDOFF.md` §8 operates), and the eval number is the policy alone, which was never subjected
to the shield at all. The gap is exactly what D15 predicts, not a mystery to explain independently of
it.

**Fix, applied 2026-09-11 (see `SESSION_HANDOFF.md` §8 for the run and numbers):** added
`--use_shield`/`--shield_type`/`--shield_alpha`/`--shield_influence_radius` flags to `evaluate.py`,
mirroring `train.py`'s, wired through to the same `build_shield_factory`. Default remains OFF, so
every existing eval number and the historical README table are unaffected and remain valid readings
of "policy alone" — they were simply never a measurement of "policy behind the shield," and should not
be described as one going forward.

---

## 2. Reference data tables

### 2.1 λ trajectory quartiles and shape, all 27 LagPPO runs

`l@X%` = λ at that fraction of logged updates. `mono` = fraction of consecutive steps where λ
increased. `S2` = second-half gain / first-half gain.

| run | l@25% | l@50% | l@75% | l@100% | mono | S2 |
|---|---|---|---|---|---|---|
| car_lagppo_shield_off_seed0 | 0.00128 | 0.00232 | 0.00309 | 0.00388 | 0.64 | 0.690 |
| car_lagppo_shield_off_seed1 | 0.00155 | 0.00268 | 0.00382 | 0.00478 | 0.69 | 0.779 |
| car_lagppo_shield_off_seed2 | 0.00114 | 0.00221 | 0.00313 | 0.00419 | 0.65 | 0.918 |
| car_lagppo_shield_on_seed0 | 0.00079 | 0.00203 | 0.00320 | 0.00420 | 0.69 | 1.081 |
| car_lagppo_shield_on_seed1 | 0.00211 | 0.00393 | 0.00555 | 0.00714 | 0.75 | 0.818 |
| car_lagppo_shield_on_seed2 | 0.00132 | 0.00294 | 0.00417 | 0.00548 | 0.71 | 0.891 |
| car_riemannian_lagppo_seed0 | 0.00101 | 0.00233 | 0.00345 | 0.00464 | 0.68 | 1.001 |
| car_riemannian_lagppo_seed1 | 0.00198 | 0.00379 | 0.00544 | 0.00639 | 0.72 | 0.687 |
| car_riemannian_lagppo_seed2 | 0.00112 | 0.00266 | 0.00384 | 0.00446 | 0.68 | 0.687 |
| goal_lagppo_shield_off_seed0 | 0.00078 | 0.00088 | 0.00115 | 0.00125 | 0.53 | 0.417 |
| goal_lagppo_shield_off_seed1 | 0.00131 | 0.00140 | 0.00163 | 0.00169 | 0.53 | 0.211 |
| goal_lagppo_shield_off_seed2 | 0.00123 | 0.00138 | 0.00162 | 0.00178 | 0.56 | 0.287 |
| goal_lagppo_shield_on_seed0 | 0.00079 | 0.00141 | 0.00189 | 0.00228 | 0.63 | 0.622 |
| goal_lagppo_shield_on_seed1 | 0.00149 | 0.00151 | 0.00172 | 0.00196 | 0.57 | 0.296 |
| goal_lagppo_shield_on_seed2 | 0.00077 | 0.00089 | 0.00083 | 0.00100 | 0.52 | 0.122 |
| goal_riemannian_lagppo_seed0 | 0.00103 | 0.00096 | 0.00106 | 0.00130 | 0.54 | 0.358 |
| goal_riemannian_lagppo_seed1 | 0.00124 | 0.00149 | 0.00161 | 0.00178 | 0.53 | 0.195 |
| goal_riemannian_lagppo_seed2 | 0.00165 | 0.00213 | 0.00227 | 0.00266 | 0.59 | 0.249 |
| lagppo_shield_off_seed0 (push) | 0.00018 | 0.00170 | 0.00564 | 0.00742 | 0.49 | 3.363 |
| lagppo_shield_off_seed1 | 0.00129 | 0.00298 | 0.00399 | 0.00462 | 0.41 | 0.553 |
| lagppo_shield_off_seed2 | 0.00106 | 0.00202 | 0.00267 | 0.00290 | 0.42 | 0.437 |
| lagppo_shield_on_seed0 | 0.00125 | 0.00243 | 0.00329 | 0.00388 | 0.40 | 0.600 |
| lagppo_shield_on_seed1 | 0.00067 | 0.00189 | 0.00330 | 0.00425 | 0.45 | 1.245 |
| lagppo_shield_on_seed2 | 0.00007 | 0.00045 | 0.00092 | 0.00162 | 0.36 | 2.584 |
| push_riemannian_lagppo_seed0 | 0.00132 | 0.00276 | 0.00289 | 0.00259 | 0.35 | −0.062 |
| push_riemannian_lagppo_seed1 | 0.00008 | 0.00017 | 0.00000 | 0.00002 | 0.26 | −0.886 |
| push_riemannian_lagppo_seed2 | 0.00030 | 0.00295 | 0.00433 | 0.00650 | 0.44 | 1.204 |

### 2.2 Full eval aggregate (all conditions, 3 seeds, from `results/eval_*/metrics.csv`)

| condition | return | per-step cost | viol rate | CVaR(0.1) |
|---|---|---|---|---|
| car_lagppo_shield_off | 28.790 ± 1.314 | 0.064 ± 0.002 | 0.90 | 178.3 |
| car_lagppo_shield_on | 28.914 ± 0.727 | 0.061 ± 0.002 | 0.92 | 151.2 |
| car_ppo_shield_off | 31.048 ± 0.165 | 0.057 ± 0.003 | 0.88 | 119.5 |
| car_ppo_shield_on | 29.271 ± 0.632 | 0.061 ± 0.012 | 0.90 | 150.0 |
| car_rcpo_shield_off | −0.082 ± 0.454 | 0.015 ± 0.004 | 0.15 | 141.2 |
| car_rcpo_shield_on | 0.025 ± 0.962 | 0.032 ± 0.024 | 0.25 | 274.5 |
| car_riemannian_lagppo | 29.596 ± 1.248 | 0.051 ± 0.006 | 0.92 | 121.3 |
| car_sac_shield_off | 37.244 ± 0.307 | 0.052 ± 0.007 | 0.85 | 114.6 |
| car_sac_shield_on | 37.178 ± 0.479 | 0.049 ± 0.002 | 0.90 | 130.2 |
| goal_lagppo_shield_off | 25.980 ± 1.120 | 0.046 ± 0.007 | 0.93 | 95.2 |
| goal_lagppo_shield_on | 26.407 ± 0.337 | 0.052 ± 0.004 | 0.92 | 107.5 |
| goal_ppo_shield_off | 26.059 ± 0.258 | 0.046 ± 0.002 | 0.87 | 95.7 |
| goal_ppo_shield_on | 26.415 ± 0.670 | 0.046 ± 0.002 | 0.87 | 114.7 |
| goal_rcpo_shield_off | −0.309 ± 1.339 | 0.019 ± 0.005 | 0.30 | 129.5 |
| goal_rcpo_shield_on | −0.412 ± 0.491 | 0.014 ± 0.009 | 0.25 | 83.3 |
| goal_riemannian_lagppo | 26.425 ± 0.154 | 0.048 ± 0.006 | 0.82 | 115.3 |
| goal_sac_shield_off | 27.500 ± 0.105 | 0.043 ± 0.003 | 0.87 | 92.7 |
| goal_sac_shield_on | 27.409 ± 0.277 | 0.048 ± 0.002 | 0.85 | 104.3 |
| push_lagppo_shield_off | 0.783 ± 0.288 | 0.050 ± 0.033 | 0.30 | 314.8 |
| push_lagppo_shield_on | 0.836 ± 0.743 | 0.035 ± 0.033 | 0.25 | 285.8 |
| push_ppo_shield_off | 1.101 ± 0.574 | 0.053 ± 0.027 | 0.32 | 375.8 |
| push_ppo_shield_on | 0.682 ± 0.207 | 0.048 ± 0.056 | 0.25 | 323.6 |
| push_rcpo_shield_off | −0.900 ± 0.548 | 0.004 ± 0.005 | 0.08 | 26.0 |
| push_rcpo_shield_on | −0.260 ± 0.454 | 0.006 ± 0.010 | 0.03 | 56.3 |
| push_riemannian_lagppo | 0.666 ± 0.250 | 0.024 ± 0.019 | 0.20 | 184.4 |
| push_sac_shield_off | 1.068 ± 0.124 | 0.021 ± 0.020 | 0.12 | 200.5 |
| push_sac_shield_on | 0.951 ± 0.144 | 0.057 ± 0.068 | 0.17 | 416.1 |

Stray directory: `results/eval_goal_riemannian_lagppo_shield_on_seed/` (n=1) is a leftover and
should be deleted or renamed.

Note `avg_len = 1000.0` and `success_rate = 0.0` across the car/goal evals — episodes never
terminate early and the goal-reach success criterion never fires.

### 2.3 Shield intervention rates (mean over 50 logged windows)

Representative values; shield_off is exactly 0.0000 everywhere, as expected.

| env | PPO on | SAC on | RCPO on | LagPPO on | Riemannian |
|---|---|---|---|---|---|
| car | 0.0200–0.0253 | 0.0226–0.0283 | 0.0219–0.0270 | 0.0198–0.0245 | 0.0198–0.0241 |
| goal | 0.0089–0.0108 | 0.0074–0.0099 | **0.0007–0.0008** | 0.0090–0.0130 | 0.0101–0.0176 |
| push | 0.0012–0.0034 | 0.0005–0.0006 | 0.0004–0.0008 | 0.0009–0.0024 | 0.0009–0.0021 |

The RCPO-vs-LagPPO 13× spread on `goal` is the D1 signature.

### 2.4 Compute reference

From `stat` mtimes on the sequential ablation logs (single RTX 3070, `num_envs=4`, 1M steps):

- PPO / RCPO / LagPPO: **43–46 min per run**
- SAC: **75–80 min per run**

---

## 3. The logging path (for whoever writes the settling script)

- **File:** `logs/<run>/metrics.jsonl`, one JSON object per line, written by
  `src/utils/logging.py:62-99` (`Logger.log_scalars`).
- **λ key:** `lagrangian/lambda`, written by `LagrangianCallback._on_step`
  (`src/algos/lagppo.py:100-117`), alongside `lagrangian/avg_cost_per_step`.
- **Cadence:** every `update_every = 2048` **env-steps summed across workers**. With `num_envs=4`
  that is one row per 512 SB3 iterations → **492 λ rows per 1M-step run**.
- **Interleaving:** those rows are mixed with **50** `TrainLoggingCallback` rows (keys `train/*`,
  cadence `eval_freq = 20000`) which carry **no λ**. Total 542 lines per run.
  **Filter on key presence, never on row index.**
- **`step` overshoots 1e6** (last row ≈ 1007616) because the dual counter and `num_timesteps`
  advance in units of `num_envs`.

**The settling classifier now exists:** `src/analysis/settling.py`, standalone, with its frozen
constants in `src/analysis/settling_thresholds.json`. Written and validated 2026-08-07.

```bash
python -m src.analysis.settling --labelled-set          # the 27 calibration runs
python -m src.analysis.settling 'logs/dualgain_*'       # read out Run 2a
python -m src.analysis.settling 'logs/cripple_*' --csv results/run3.csv
```

It deliberately exposes no way to tune thresholds from the command line.

**Naming seam, and a trap it contains.** Training logs are `logs/<tag>`, evals are
`results/eval_<tag>` — except the original Push runs, which are logged as
`logs/lagppo_shield_off_seed0` but evaluated into `results/eval_push_lagppo_shield_off_seed0`.
A suffix match is **not** a safe bridge: `car_lagppo_shield_off_seed0` also ends with
`lagppo_shield_off_seed0`, so a naive `eval_*<tag>` glob silently supplies the *car's* reward scale
for the *push* run. That bug was hit and fixed during development; `find_eval_metrics` now bridges
the seam with an explicit env-prefix rule and treats residual ambiguity as an error rather than
guessing. Anything else that maps logs to results needs the same care.

Prior art, for reference only: `src/plot_three_way.py` (`extract_series`, `LOG_PATTERNS`,
`SMOOTH_WINDOW = 20`). It has a **duplicate `("push","none")` dict key at lines 49-50**, the first
of which is silently dead. `settling.py` does not import from it.

---

## 4. Step 0 — the frozen settling metric

### 4.1 Definitions (to be frozen before any new data exists)

- **S1 (primary).** Window = final 20% of `step`. OLS line on λ vs step.
  `S1 = slope · window_length / mean(λ over window)`.
- **S2 (primary, co-equal).** Deceleration ratio `g₂/g₁`, where `g₁ = λ(0.5T) − λ(0)` and
  `g₂ = λ(T) − λ(0.5T)`.
- **S3 (dispersion).** `std(residuals about the S1 line) / |mean(λ over window)|`.
- **S4 (feasibility guard).** Mean windowed `lagrangian/avg_cost_per_step`, and the fraction of
  windows above budget, over the same window.
- **S5 (dual-authority guard).**
  `max(λ) · per_step_cost / (avg_return / avg_len)`, with the reward-scale terms taken from the
  run's `results/eval_*/metrics.csv` (the converged policy's reward scale) — the fraction of the
  per-step reward signal that the penalty actually represents. This is the statistic that would
  have caught D5.

  **Measured on the labelled set (2026-08-07, `results/settling_labelled_set.csv`):**

  | env | S5 range across 9 LagPPO runs | reading |
  |---|---|---|
  | CarGoal | 0.007 – 0.015 | dual unambiguously open; λ worth ~1% of the reward signal |
  | PointGoal | 0.002 – 0.005 | dual unambiguously open |
  | PointPush | 0.003 – 0.925 | λ genuinely has authority in some seeds, none in others |

  **23 of 27 runs are VOID at the 0.20 threshold.** The four that are not are all PointPush.

  This is a sharper result than "λ is inert everywhere." **PointPush is the only environment where
  the multiplier ever had authority** — and PointPush is exactly where the preprint's central
  complementarity finding comes from. So that finding is not fabricated; it rests on three seeds
  whose dual authority varies by a factor of ~50 (0.017 to 0.925). The car anomaly, by contrast,
  lives entirely in the S5 ≈ 0.01 regime, which is what makes the open-dual-loop reading so strong
  there specifically.

### 4.2 Proposed thresholds (PENDING SIGN-OFF — see §7)

Calibrated on the labelled set (car = not-settled, PointGoal = settled). PointPush excluded as
unlabelled: it is inconsistent in both directions and must not drive thresholds.

- **Settled:** S2 ≤ 0.40 **and** S1 ≤ 0.10. (Labelled: 6/9 goal pass, 0/9 car.)
- **Not settled / persistent climb:** S2 ≥ 0.65. (Labelled: 9/9 car, 0/9 goal.)
- **Limit cycle:** S1 ≤ 0.10 with S3 ≥ 0.05.
- **Undetermined:** anything else — reported, not hidden.
- **Void:** S5 < 0.20 ⇒ run excluded from settling classification and reported as an open-loop dual.

### 4.3 Disclosure that must appear in the paper

S1 was specified before looking at data. **S2 was added as a form after inspecting the labelled
set**, because S1 does not separate car from PointGoal at all (D6) whereas S2 separates 9/9 with
zero overlap. This is legitimate calibration only because (a) the labelled set is disjoint from
every run in the new program, (b) both statistics are frozen before any new data exists, and (c) it
is disclosed. Both S1 and S2 are reported for every seed regardless of which one moves.

The continuous statistics are the primary reported quantity; the settled/not-settled binary is a
secondary summary table only.

---

## 5. The execution program

### Run 0 (NEW, BLOCKING) — make the shield the object the paper describes

Runs 1–4 all read out through λ and through the shield. λ currently carries no signal (D5) and the
shield is not the object the paper describes (D1–D4). Running the 48-run factorial on this code
would spend ~36 GPU-hours measuring a shield that reacts to accelerometer noise against an episode-0
hazard field, read out through a multiplier that cannot move the policy.

Stories 2 and 3 are untestable as stated until this is fixed: story 2 ("the shield asks for a
one-step deflection the car cannot execute") presupposes the shield is asking for a hazard-avoiding
deflection at all.

**0a. Accessor probe (~30 min, unblocks Run 0 AND Run 3).** A short script on the WSL box that
prints available `env.unwrapped` accessors for agent position and heading, and asserts position
changes under motion. Candidates: `env.unwrapped.task.agent.pos` / `.mat`, MuJoCo
`data.body(...).xpos` / body quaternion → yaw.

**0b. The four fixes.**

1. **Position.** Replace `_extract_agent_xy` (`shield.py:94-125`). Preferred approach: have
   `ShieldingActionWrapper.step` (`make_env.py:288-290`) supply the position directly from
   `env.unwrapped` rather than passing `obs`. Keeps the shield a pure geometric function and makes
   the dependency explicit in the wrapper that already owns `_last_obs`.
2. **Hazards per episode.** Move the introspection block (`train.py:238-266`) out of `factory` into
   a `refresh_hazards(env)` method, called from `ShieldingActionWrapper.reset`
   (`make_env.py:314-319`) after `env.reset()` returns.
3. **Riemannian sign + saturation.** Flip line 198 to `a[:2] + deflection` **or** negate `grad_i` at
   line 147 — one or the other, not both. Replace element-wise `np.clip` with a norm-based clip so
   direction is preserved: `n = ||g||; deflection = alpha * g * min(1, max_norm/n)`. This changes
   what `alpha=0.1` means; a quick alpha re-scan is needed.
4. **Decide D4 deliberately.** Either implement Eq. (5) (tangential projection) and keep the paper
   text, or keep bisection scaling and rewrite §3.3 / §4.3 / A.3.
   **Recommendation: keep bisection, fix the prose** — it is the more conservative operator and the
   one with 27 runs of accumulated intuition.

**Acceptance gate.** A 50k-step probe on `SafetyPointGoal1-v0` where:
(i) logged shield position matches `env.unwrapped` ground truth to 1e-6;
(ii) hazard centers change across episode boundaries;
(iii) intervention rate rises materially above the current 0.009;
(iv) a pure-geometry assertion that no shielded action's predicted next position lies inside any
hazard.

### Run 1 (car, shield off) — ALREADY DONE AT 3 SEEDS; THE ANSWER IS ALREADY IN

**What changes:** drop `--use_shield`. Nothing else. `train.py:408` passes
`use_shield=args.use_shield`; `make_env.py:402` skips the wrapper.
**Existing runs:** `logs/car_lagppo_shield_off_seed{0,1,2}`.

**Result:** S1 = 0.180/0.138/0.191 shield-off vs 0.244/0.186/0.237 shield-on; S2 = 0.69/0.78/0.92
off vs 1.08/0.82/0.89 on. **The multiplier misbehaves identically without the shield.** Run 1 as
specified does **not** exclude story 1 — it is currently the strongest evidence *for* it.

Must be re-run post-Run-0 (the shield-on comparator changes) and extended to 8 seeds, but this is
how it currently reads and it should be known going into GRAIL.

### Run 2 (PI-Lagrangian)

**Why a flag won't do it.** `LagrangianState.update` (`lagppo.py:39-53`) is:

```python
delta = self.lr_lambda * (avg_cost_per_step - self.cost_budget)
self.lam = max(0.0, min(self.clip_lam_max, self.lam + delta))
```

The integral state **is** `self.lam`. P cannot be bolted on without restructuring: λ must be
*derived* from a separate accumulator plus a proportional term, rather than *being* the accumulator.

**Minimal restructure:**

```
err      = avg_cost - budget
self.I   = self.I + lr_lambda * err          # integral accumulator, UNPROJECTED
lam_raw  = self.I + kp * err                 # P + I combined
self.lam = clip(lam_raw, 0, clip_lam_max)    # projection AFTER the sum
```

Projection applies to `lam_raw` only. `self.I` is deliberately **not** projected — that is what lets
a negative-error window be remembered instead of clipped away, and it is precisely the difference
from per-term projection. `kp = 0.0` reproduces the current update exactly, provided `I` and `lam`
both start at 0. **Make that a unit test** alongside `src/tests/test_lagrangian_update.py`.

**Sweep:** `kp ∈ {0, 1e-3, 1e-2, 1e-1}` (3 nonzero, log-spaced), integral gain held at the paper's
`lr_lambda = 5e-4`. New CLI flag `--kp_lambda`, threaded through `train.py:507-512`.

**Positive control — RUN THIS FIRST (phase 2a).** Given D5, the honest positive control is not a
`kp` value but a run where λ demonstrably moves the policy. Add an `lr_lambda = 5e-2` arm (100×, the
scale from D5 needed to reach λ ≈ 0.45). If λ at 100× dual gain still climbs monotonically with cost
above budget, the dual-dynamics story is genuinely dead and the geometric claim is earned. If it
settles, D5 was the whole anomaly. **This single 8-run arm (~6 h) is worth more than the entire kp
sweep.**

**Budget/feasibility axis.** Add `d ∈ {0.05, 0.10}` on the `kp=0` cell only. Note the competing
explanation already in hand: `car_sac_shield_off` reaches 0.052 (shield-on 0.049), so 0.05 is not
obviously below the car's achievable floor.

### Run 3 (crippled point robot, SafetyPointGoal1-v0)

**Wrapper.** A `gym.ActionWrapper` decomposing `a_xy` relative to the robot's current heading θ:

```
u  = [cos θ, sin θ]
a∥ = (a · u) u
a⊥ = a − a∥
a' = a∥ + alpha · a⊥
```

**Hook point.** `make_env.py:394-406`, inserted immediately after `_try_make` — i.e. **innermost,
below everything**:

```
base → CrippleWrapper → CostInfoWrapper → RewardShapingWrapper → ShieldingActionWrapper → Monitor
```

Actions flow inward, so the shield's **corrected** action is what gets crippled. That is exactly the
required semantics: the shield requests a lateral deflection the platform cannot execute. Placing it
below `CostInfoWrapper` also leaves cost/step-signature normalization untouched.

New `make_env` parameter `cripple_alpha: Optional[float] = None`; new CLI flag `--cripple_alpha`;
threaded at `train.py:405-425`.

**Heading source.** Not in the 60-dim observation. Must come from `env.unwrapped` (MuJoCo body
quaternion → yaw). Same blocking unknown as Run 0a — resolve once, both runs depend on it.

**Null test (alpha=1).** The formula gives `a' = a∥ + a⊥ = a` — an algebraically **exact** identity,
not an approximation. Assert bitwise equality in a unit test, **and** verify the 8-seed alpha=1 arm
reproduces the post-Run-0 PointGoal LagPPO result within seed noise **before looking at anything
else**.

Important: "my existing PointGoal1 settling result" is not currently a settling result (D6) and will
change after Run 0. The null test compares against the **post-Run-0** baseline — which means the
alpha=1 shield-on cell *is* that baseline.

**Design.** 2×3×8 = **48 runs**: shield ∈ {Riemannian, off} × alpha ∈ {1, 0.5, 0} × 8 seeds.
Geometric shield dropped from this causal experiment (it is known to degrade on dense hazard fields,
reintroducing the confound being removed). Headline claim is the **interaction**: settling degrades
as alpha falls under the shield, and degrades less or not at all without it. Power concentrated at
the alpha=1 and alpha=0 endpoints; alpha=0.5 establishes monotonicity only.

**Launcher.** The existing scripts (`run_ablation_cargoal.sh` etc.) are fully unrolled — 24
copy-pasted `python -m src.train` blocks. That does not scale to 48. Write one
`run_cripple_factorial.sh` with three nested loops emitting `logs/cripple_{shield}_a{alpha}_seed{s}`
and matching `checkpoints/cripple/...`, plus a `--dry-run` that prints commands.

### Run 4 (representation) — LAST, and only if Run 3's interaction is real

Fix `alpha=0`, shield on, seeds identical; change only the features the policy sees. Two arms
(body-frame chart; appended heading-aligned basis) × 8 seeds = 16 runs. Needs an **observation**
wrapper — a new class, no existing hook — at the same innermost position. Moot if Run 3 is null.

---

## 6. Sequencing, seeds, compute

| phase | runs | compute | gate |
|---|---|---|---|
| 0a. Accessor probe | — | ~0.5 h | accessor names confirmed |
| 0b. D1–D4 fixes + unit tests + 50k probe | — | ~2 h | Run 0 acceptance gate |
| 0c. Re-run 3-env × 3-shield LagPPO, 8 seeds | 72 | ~54 h | new baseline + frozen-metric readout |
| 1. Car shield-off, 8 seeds | — | — | folded into 0c |
| 2a. Dual-authority control (`lr_lambda` ×100) | 8 | ~6 h | **decides whether 2b/3/4 are worth running** |
| 2b. PI sweep kp ∈ {0,1e-3,1e-2,1e-1} × 8 | 32 | ~24 h | |
| 2c. Budget axis d=0.10, kp=0, 8 seeds | 8 | ~6 h | |
| 3. Cripple factorial 2×3×8 | 48 | ~36 h | alpha=1 null test FIRST |
| 4. Representation 2×8 | 16 | ~12 h | only if Run 3 interaction is real |

**Total ≈ 145 h ≈ 6 days** of continuous compute plus implementation. Fits an autumn 2026 deadline
with room, but only if Run 0 starts promptly.

**Seed decision: 8 seeds everywhere, including re-runs of existing conditions.** Rationale: (a)
post-Run-0 the old runs are invalid anyway, so consistency costs nothing extra; (b) preprint §5.5
already concedes nothing but the λ separation survives 3 seeds — and that λ separation is 3-vs-3
with fully overlapping ranges (off: 0.0029/0.0046/0.0074; on: 0.0016/0.0039/0.0043). Phase 0c at 8
seeds fixes the single most-cited statistical weakness in the paper.

**Ordering note.** Phase 2a goes **before** the PI sweep. It is 6 hours and the highest-information
run in the program: it directly tests whether the anomaly is the open dual loop (D5). If λ at 100×
gain settles the car, Runs 2b/3/4 are answering a question that no longer exists — and the paper's
story changes to something different but still publishable and much better supported.

---

## 7. Decisions — RESOLVED 2026-08-07

1. **Run 0 happens**, but Run 2a launches first and in parallel. Run 2a does not depend on Run 0:
   it is shield-off, so D1–D4 cannot touch it, and it needs no code changes. Launcher:
   `run_dualgain_car.sh`.
2. **Run 0 scope: fix D1–D4 (and D7) now; defer the 72-run re-baseline** until Run 2a reports. If
   2a shows the anomaly was the open dual loop, the re-baseline is a much smaller job than 72 runs.
3. **D4: rewrite the prose to match bisection.** `shield.py` keeps the bisection operator; preprint
   §3.3, §4.3 and A.3 (incl. Eq. 5) get rewritten. No existing results change from this decision.
4. **§4.2 thresholds approved as written** and frozen in
   `src/analysis/settling_thresholds.json`. S2 and S5 are pre-registered statistics, carrying the
   §4.3 disclosure that S2's form was chosen on the labelled set.
5. **8 seeds everywhere**, but the 54 h re-baseline waits on 2a's result (see decision 2).

### Immediate state

| item | status |
|---|---|
| `run_dualgain_car.sh` | written, ready to launch, resumable |
| `src/analysis/settling.py` + `settling_thresholds.json` | written, validated against the labelled set |
| `results/settling_labelled_set.csv` | generated — the frozen calibration readout |
| `scripts/probe_env_accessors.py` | written, **must be run on WSL before Run 0 can proceed** |
| D1–D4, D7 fixes | blocked on the probe |

### Run 2a decision table

| outcome | reading | consequence |
|---|---|---|
| λ → ~0.4, cost drops below budget, S2 < 0.40 | D5 was the whole anomaly | paper becomes "my constrained method was silently unconstrained"; Runs 3–4 unnecessary |
| λ → ~0.4, cost stays above budget | dual dynamics genuinely excluded | geometric claim earned; the full ~145 h is worth spending |
| λ overshoots and oscillates | limit cycle | the PI sweep becomes the headline run, not a falsification chore |

---

## 8. Preprint overclaim ledger

| Preprint claim | Status |
|---|---|
| §4.1 "observation space ... encoding the agent's position and velocity, the box and goal positions" | **False.** 76-dim sensor + 4×16 lidar; no absolute position. This is the false premise behind D1. |
| §4.3, A.3 "hazard positions ... read ... at the start of each episode" | **False.** Read once at construction (D2). |
| §3.3, §4.3, A.3 Eq. (5) tangential projection, "closest safe action" | **Not implemented.** Code does isotropic bisection scaling (D4). Not the closest safe action under either operator. |
| §5.3 "Both conditions show monotonically increasing λ" | Step-to-step monotone fraction 0.36–0.49 on push. Trend increases; "monotonically" is wrong. |
| §5.3 "the central finding" — shield reduces λ growth, 0.005 vs 0.003 | Numbers reproduce, but n=3 with fully overlapping ranges, and λ ≤ 0.0075 throughout (D5). At that magnitude the multiplier does not measurably influence the policy, so "the Lagrangian has to work less hard" is an interpretation the data cannot carry. |
| §5.2 "PPO shield-on has the highest intervention rate ... unconstrained algorithms rely most heavily on the shield" | Under D1 the ordering is a function of accelerometer statistics, not hazard proximity. Not defensible until Run 0. |
| §6.1 complementarity / different timescales | Framing is fine; empirical support is a 3-vs-3 comparison of an inert multiplier under a shield that could not see hazards. |
| Abstract "adding a shield measurably reduces the growth of the Lagrange multiplier" | "Measurably" is doing too much work at n=3 with overlapping ranges. Testable properly after phase 0c. |
| README l.33 / report §11 "point-robot tasks λ rises and then plateaus ... CarGoal climbs the full million steps" | **Not supported by the logs** (D6). No point run plateaus. Survivable version: PointGoal decelerates (S2 ≤ 0.63, 9/9), Car does not (S2 ≥ 0.68, 9/9). |
| Report §11 "achievable cost floor ~0.051 > budget 0.05 ⇒ infeasible" | 0.051 is Riemannian *eval* cost, not a floor. `car_sac_shield_off` reaches 0.052 / shield-on 0.049 with return 37.2. Infeasibility not established. |

**The single most important thing to internalize before GRAIL:** as the code stands, the car
anomaly's best-supported explanation is story 1, in the specific form *"the dual gain is ~100× too
small to close the loop"* — and it applies to all three environments equally. The likely first
question about a multiplier that rises forever is "what is λ actually worth relative to the reward?"
and the current answer is ~1%.

This is a **better** anomaly than the current one. *"I found that my constrained method was silently
unconstrained, here is the scale argument that proves it, and here is the controlled sweep that
separates it from the geometric explanation"* is a stronger story than the one in the preprint, and
phase 2a decides it in six hours.

---

## Appendix A — Diagnostic script used for §1, §2.1

Read-only. Re-runnable with stdlib + numpy only (no torch / SB3 / safety-gymnasium needed).

```python
"""Summarize logged lambda trajectories. Usage: python peek_lambda.py <branch_dir>"""
import json, os, sys, glob
import numpy as np

ROOT = sys.argv[1]
for d in sorted(glob.glob(os.path.join(ROOT, "logs", "*"))):
    p = os.path.join(d, "metrics.jsonl")
    if not os.path.isfile(p):
        continue
    steps, lam, cost = [], [], []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "lagrangian/lambda" in r:                      # filter on KEY, never on index
            steps.append(r["step"])
            lam.append(r["lagrangian/lambda"])
            cost.append(r.get("lagrangian/avg_cost_per_step", float("nan")))
    if len(lam) < 50:
        continue
    steps, lam, cost = map(lambda x: np.asarray(x, float), (steps, lam, cost))

    # S1 / S3 over the final 20% window
    cut = steps[-1] - 0.2 * (steps[-1] - steps[0])
    m = steps >= cut
    sw, lw, cw = steps[m], lam[m], cost[m]
    slope, intercept = np.linalg.lstsq(
        np.vstack([sw, np.ones_like(sw)]).T, lw, rcond=None)[0]
    resid = lw - (slope * sw + intercept)
    S1 = slope * (sw[-1] - sw[0]) / lw.mean()
    S3 = resid.std() / abs(lw.mean())

    # S2 deceleration ratio
    half = lam[int(0.5 * (len(lam) - 1))]
    g1, g2 = half - lam[0], lam[-1] - half
    S2 = g2 / g1 if g1 else float("nan")

    print(f"{os.path.basename(d):<34} S1={S1:>8.4f} S2={S2:>8.3f} S3={S3:>7.4f} "
          f"lam_max={lam.max():.5f} cost_win={cw.mean():.4f} "
          f"frac_over={(cw > 0.05).mean():.2f} mono={(np.diff(lam) > 0).mean():.2f}")
```

## Appendix B — How the observation dims were obtained

No torch or gymnasium required; the shapes are recoverable straight from the pickle opcodes.

```python
import zipfile, json, base64, pickletools, io

for name, p in [("push", "checkpoints/push/lagppo_shield_on/seed_0/latest.zip"),
                ("goal", "checkpoints/goal/lagppo_shield_on/seed_0/latest.zip"),
                ("car",  "checkpoints/car/lagppo_shield_on/seed_0/latest.zip")]:
    d = json.loads(zipfile.ZipFile(p).read("data"))
    for k in ("observation_space", "action_space"):
        s = io.StringIO()
        pickletools.dis(base64.b64decode(d[k][":serialized:"]), s)
        txt = s.getvalue()
        i = txt.find("'_shape'")
        first_int = next(l.strip() for l in txt[i:i + 300].split("\n") if "BININT" in l)
        print(name, k, first_int)
```

Output: push 76/2, goal 60/2, car 72/2.
