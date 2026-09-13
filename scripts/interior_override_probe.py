"""
Test the full-authority interior override against the "current numbers"
(measurement A in SESSION_HANDOFF.md: car entries 0.95-1.15/ep, dwell
654-810; point entries 3.15-3.55/ep, dwell 15-19).

Settles whether the interior trap (SESSION_HANDOFF.md's escape-dynamics
finding: the geometric bisection is mathematically incapable of returning a
nonzero action scale once already inside a hazard, regardless of kinematic
model) is a consequence of the shield BLENDING a weak/zeroed correction with
the policy's own action, or is inherent to action-space filtering itself.
`interior_override=True` (src/safety/shield.py) replaces -- does not blend --
the action with a full-magnitude escape command whenever the agent is
already inside a hazard, using the fitted heading-relative model (Part 1) to
choose the escape direction.

Uses the EXISTING trained `*_riemannian_postfix` checkpoints (no retraining:
this is a deployment-time shield change, same policy either way), same
20-episode/seed=100/deterministic protocol as every other shielded-eval
measurement in this project, alpha=0.1/influence_radius=0.4 unchanged,
kinematic_model="world_xy" unchanged (the override is orthogonal to that
choice -- see SESSION_HANDOFF.md's Part 2 causal-test writeup) -- only
interior_override toggles.

Usage:
    .venv/bin/python -m scripts.interior_override_probe
"""

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.envs.make_env import make_env, resolve_hazards
from src.train import build_shield_factory

ENV_IDS: Dict[str, str] = {"car": "SafetyCarGoal1-v0", "goal": "SafetyPointGoal1-v0"}
SEEDS = (0, 1, 2)


def checkpoint_path(robot: str, seed: int) -> str:
    return f"checkpoints/{robot}_riemannian_postfix/lagppo_shield_on/seed_{seed}/latest.zip"


def clearance_to_nearest(pos: np.ndarray, hazards: List[Tuple[float, float, float]]) -> float:
    return min(float(np.linalg.norm(pos - np.array([h[0], h[1]]))) - h[2] for h in hazards)


def run_episode(env: Any, model: PPO, seed: int) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    hazards = resolve_hazards(env)
    clearances: List[float] = []
    ep_ret, ep_cost = 0.0, 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        pos = np.asarray(info["shield_agent_pos"], dtype=float)
        clearances.append(clearance_to_nearest(pos, hazards))
        ep_ret += float(r)
        ep_cost += float(info.get("cost", 0.0))
        done = bool(terminated or truncated)
    return {"clearances": clearances, "return": ep_ret, "cost": ep_cost, "len": len(clearances)}


def entries_and_dwells(clearances: List[float]) -> Tuple[int, List[int], List[bool]]:
    """Same definition as scripts/entry_dwell_probe.py, plus per-run 'escaped' flags."""
    entries = 0
    dwells: List[int] = []
    escaped: List[bool] = []  # True if the run ended by leaving the hazard, not by episode end
    prev_inside = False
    dwell = 0
    n = len(clearances)
    for i, c in enumerate(clearances):
        cur_inside = c <= 0.0
        if cur_inside and not prev_inside:
            entries += 1
            dwell = 1
        elif cur_inside:
            dwell += 1
        elif prev_inside and not cur_inside:
            dwells.append(dwell)
            escaped.append(True)
            dwell = 0
        prev_inside = cur_inside
    if prev_inside:
        dwells.append(dwell)
        escaped.append(False)  # still inside when the episode ended: censored, did not escape
    return entries, dwells, escaped


def evaluate_config(env_id: str, checkpoint: str, episodes: int, seed: int,
                     alpha: float, influence_radius: float, interior_override: bool) -> None:
    shield_factory = build_shield_factory(
        env_id, shield_type="riemannian", alpha=alpha, influence_radius=influence_radius,
        kinematic_model="world_xy", interior_override=interior_override,
    )
    env = make_env(env_id, seed=seed, use_shield=True, shield_factory=shield_factory)
    model = PPO.load(checkpoint)

    all_dwells: List[int] = []
    all_escaped: List[bool] = []
    total_entries = 0
    total_inside_steps = 0
    total_steps = 0
    returns: List[float] = []
    per_step_costs: List[float] = []

    for ep in range(episodes):
        result = run_episode(env, model, seed=seed + ep)
        entries, dwells, escaped = entries_and_dwells(result["clearances"])
        total_entries += entries
        all_dwells.extend(dwells)
        all_escaped.extend(escaped)
        total_inside_steps += sum(1 for c in result["clearances"] if c <= 0.0)
        total_steps += result["len"]
        returns.append(result["return"])
        per_step_costs.append(result["cost"] / result["len"])

    env.close()

    escaped_dwells = [d for d, e in zip(all_dwells, all_escaped) if e]
    n_escaped = sum(all_escaped)
    n_total_runs = len(all_dwells)

    tag = "interior_override=ON " if interior_override else "interior_override=OFF"
    print(f"  [{tag}] entries/ep={total_entries/episodes:.2f}  "
          f"median_dwell={np.median(all_dwells) if all_dwells else 0:.1f}  "
          f"max_dwell={int(np.max(all_dwells)) if all_dwells else 0}  "
          f"inside_frac={total_inside_steps/total_steps:.3f}")
    print(f"       trapped runs: {n_total_runs}, of which ESCAPED (not censored to episode end): "
          f"{n_escaped}/{n_total_runs}")
    if escaped_dwells:
        print(f"       of the ones that escaped: steps-to-escape median={np.median(escaped_dwells):.1f} "
              f"mean={np.mean(escaped_dwells):.1f} min={np.min(escaped_dwells)} max={np.max(escaped_dwells)}")
    print(f"       return: mean={np.mean(returns):.3f} (per-episode: {[round(r,2) for r in returns[:5]]}...)")
    print(f"       per_step_cost: mean={np.mean(per_step_costs):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--influence_radius", type=float, default=0.4)
    args = ap.parse_args()

    for robot, env_id in ENV_IDS.items():
        for seed in SEEDS:
            checkpoint = checkpoint_path(robot, seed)
            print(f"\n{'='*70}\n{robot} seed={seed} ({env_id})\n{'='*70}")
            for interior_override in (False, True):
                evaluate_config(env_id, checkpoint, args.episodes, args.seed,
                                 args.alpha, args.influence_radius, interior_override)


if __name__ == "__main__":
    main()
