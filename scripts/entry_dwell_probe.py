"""
Measure hazard entry events and dwell time under shielded deployment.

Reproduces the methodology of SESSION_HANDOFF.md's 2026-09-11 measurement A
(entries/episode, mean/median/max dwell, inside-step fraction) for arbitrary
checkpoints/shield configs, so it can be reused as the readout for the
kinematic-model causal test (run_kinematic_model_causal_test.sh): does
switching the shield's next-position prediction from "world_xy" (current,
near-zero direction cosine with true displacement -- see
scripts/fit_kinematic_model.py) to "heading_fit" (data-fit, high cosine)
change entry prevention or dwell?

An "entry" is a step where clearance (distance to the nearest hazard center,
minus that hazard's radius) is <= 0, immediately following a step where it
was > 0 (or the first step of an episode, if it starts already inside).
"Dwell" is the run-length of consecutive inside-hazard steps starting from an
entry.

Usage:
    .venv/bin/python -m scripts.entry_dwell_probe
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.envs.make_env import make_env, resolve_hazards
from src.train import build_shield_factory


@dataclass
class RunConfig:
    label: str
    env_id: str
    checkpoint: str
    kinematic_model: str


# The 4 causal-test checkpoints (run_kinematic_model_causal_test.sh, 150k
# steps, seed 0), plus the two existing full-1M riemannian_postfix
# checkpoints (world_xy only) for context against the historical numbers.
RUNS: List[RunConfig] = [
    RunConfig("car/world_xy (150k, causal test)", "SafetyCarGoal1-v0",
              "checkpoints/car_kinmodel_world_xy/lagppo_shield_on/seed_0/latest.zip", "world_xy"),
    RunConfig("car/heading_fit (150k, causal test)", "SafetyCarGoal1-v0",
              "checkpoints/car_kinmodel_heading_fit/lagppo_shield_on/seed_0/latest.zip", "heading_fit"),
    RunConfig("goal/world_xy (150k, causal test)", "SafetyPointGoal1-v0",
              "checkpoints/goal_kinmodel_world_xy/lagppo_shield_on/seed_0/latest.zip", "world_xy"),
    RunConfig("goal/heading_fit (150k, causal test)", "SafetyPointGoal1-v0",
              "checkpoints/goal_kinmodel_heading_fit/lagppo_shield_on/seed_0/latest.zip", "heading_fit"),
]


def clearance_to_nearest(pos: np.ndarray, hazards: List[Tuple[float, float, float]]) -> float:
    """Signed distance to the nearest hazard boundary (<=0 means inside)."""
    best = float("inf")
    for (hx, hy, hr) in hazards:
        d = float(np.linalg.norm(pos - np.array([hx, hy])))
        best = min(best, d - hr)
    return best


def run_episode(env: Any, model: PPO, seed: int) -> Dict[str, Any]:
    """Roll out one deterministic episode and track the clearance trace."""
    obs, _ = env.reset(seed=seed)
    hazards = resolve_hazards(env)
    clearances: List[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        pos = np.asarray(info["shield_agent_pos"], dtype=float)
        clearances.append(clearance_to_nearest(pos, hazards))
        done = bool(terminated or truncated)
    return {"clearances": clearances}


def entries_and_dwells(clearances: List[float]) -> Tuple[int, List[int]]:
    """
    Count entry events and their dwell lengths from a clearance trace.

    :param clearances: Per-step signed clearance to the nearest hazard.
        :type clearances: List[float]

    :return: (n_entries, list of dwell lengths, one per entry).
        :rtype: Tuple[int, List[int]]
    """
    entries = 0
    dwells: List[int] = []
    inside = False
    dwell = 0
    prev_inside = False
    for c in clearances:
        cur_inside = c <= 0.0
        if cur_inside and not prev_inside:
            entries += 1
            dwell = 1
        elif cur_inside:
            dwell += 1
        elif prev_inside and not cur_inside:
            dwells.append(dwell)
            dwell = 0
        prev_inside = cur_inside
    if prev_inside:
        dwells.append(dwell)
    return entries, dwells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--influence_radius", type=float, default=0.4)
    args = ap.parse_args()

    header = (f"{'run':<38} {'entries/ep':>10} {'mean dwell':>10} {'med dwell':>9} "
              f"{'max dwell':>9} {'inside frac':>11}")
    print(header)
    print("-" * len(header))

    for cfg in RUNS:
        shield_factory = build_shield_factory(
            cfg.env_id, shield_type="riemannian", alpha=args.alpha,
            influence_radius=args.influence_radius, kinematic_model=cfg.kinematic_model,
        )
        env = make_env(cfg.env_id, seed=args.seed, use_shield=True, shield_factory=shield_factory)
        model = PPO.load(cfg.checkpoint)

        all_entries = 0
        all_dwells: List[int] = []
        all_inside_steps = 0
        all_steps = 0

        for ep in range(args.episodes):
            result = run_episode(env, model, seed=args.seed + ep)
            entries, dwells = entries_and_dwells(result["clearances"])
            all_entries += entries
            all_dwells.extend(dwells)
            all_inside_steps += sum(1 for c in result["clearances"] if c <= 0.0)
            all_steps += len(result["clearances"])

        env.close()

        entries_per_ep = all_entries / args.episodes
        mean_dwell = float(np.mean(all_dwells)) if all_dwells else 0.0
        med_dwell = float(np.median(all_dwells)) if all_dwells else 0.0
        max_dwell = int(np.max(all_dwells)) if all_dwells else 0
        inside_frac = all_inside_steps / all_steps if all_steps else 0.0

        print(f"{cfg.label:<38} {entries_per_ep:>10.2f} {mean_dwell:>10.1f} {med_dwell:>9.1f} "
              f"{max_dwell:>9d} {inside_frac:>11.3f}")


if __name__ == "__main__":
    main()
