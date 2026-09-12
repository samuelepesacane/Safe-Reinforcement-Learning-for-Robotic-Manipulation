"""
Instrument the escape asymmetry: point recovers from a hazard entry in a
median 15-19 steps, car in a median 654-810 (SESSION_HANDOFF.md measurement
A) -- a ~40x gap with no mechanism attached yet. This script logs, per step,
the agent's world-frame velocity vector (env.unwrapped.task.agent.vel) and
heading (env.unwrapped.task.agent.mat), and for every trapped run (a maximal
sequence of consecutive steps with clearance <= 0 to the nearest hazard)
measures:

  - time to reverse velocity direction: steps from entry until the velocity's
    component along the outward radial direction (from the trapping hazard's
    center, fixed at the direction observed at entry) first becomes positive
    -- i.e. the agent is actually moving away, not just commanded to.
  - mean speed during the trapped run, and whether it decays (inertia: the
    agent cannot decelerate/reverse quickly) or stays high while pointed the
    wrong way (steering: direction is the bottleneck, not speed).
  - total heading rotation during the trapped run (steering activity).

Uses the actual trained `*_riemannian_postfix` checkpoints (full 1M steps),
the actual shield (alpha=0.1, influence_radius=0.4), the same 20-episode/
seed=100/deterministic protocol as the rest of this project's shielded-eval
measurements, for direct comparability.

Usage:
    .venv/bin/python -m scripts.escape_dynamics_probe
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.envs.make_env import make_env, resolve_hazards
from src.train import build_shield_factory

CHECKPOINTS: Dict[str, Tuple[str, str]] = {
    "car": ("SafetyCarGoal1-v0",
            "checkpoints/car_riemannian_postfix/lagppo_shield_on/seed_0/latest.zip"),
    "goal": ("SafetyPointGoal1-v0",
             "checkpoints/goal_riemannian_postfix/lagppo_shield_on/seed_0/latest.zip"),
}


@dataclass
class TrapRun:
    """One maximal trapped run (consecutive clearance<=0 steps)."""

    dwell: int
    time_to_reverse: Optional[int]  # None if velocity never points outward within the run
    mean_speed: float
    speed_first_half: float
    speed_second_half: float
    total_heading_rotation: float  # radians, sum of |wrapped delta| over the run
    censored: bool = field(default=False)  # run ends at episode end, not by escaping


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def get_heading(env: Any) -> float:
    mat = np.asarray(env.unwrapped.task.agent.mat, dtype=float).reshape(3, 3)
    return math.atan2(mat[1, 0], mat[0, 0])


def get_velocity_xy(env: Any) -> np.ndarray:
    return np.asarray(env.unwrapped.task.agent.vel, dtype=float)[:2]


def nearest_hazard(pos: np.ndarray, hazards: List[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], float]:
    best_h, best_c = None, float("inf")
    for h in hazards:
        d = float(np.linalg.norm(pos - np.array([h[0], h[1]]))) - h[2]
        if d < best_c:
            best_c, best_h = d, h
    return best_h, best_c


def run_episode(env: Any, model: PPO, seed: int) -> Dict[str, List]:
    """Roll out one deterministic episode, logging pos/heading/vel/clearance."""
    obs, _ = env.reset(seed=seed)
    hazards = resolve_hazards(env)
    pos_trace, heading_trace, vel_trace, clear_trace, nearest_trace = [], [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        heading = get_heading(env)  # heading BEFORE the step, matches pos-before semantics
        obs, _, terminated, truncated, info = env.step(action)
        pos = np.asarray(info["shield_agent_pos"], dtype=float)
        vel = get_velocity_xy(env)  # velocity AFTER the step (resulting motion)
        h, c = nearest_hazard(pos, hazards)
        pos_trace.append(pos)
        heading_trace.append(heading)
        vel_trace.append(vel)
        clear_trace.append(c)
        nearest_trace.append(h)
        done = bool(terminated or truncated)
    return dict(pos=pos_trace, heading=heading_trace, vel=vel_trace,
                clearance=clear_trace, nearest=nearest_trace)


def extract_trap_runs(trace: Dict[str, List]) -> List[TrapRun]:
    """Find maximal trapped runs and compute escape-dynamics metrics for each."""
    n = len(trace["clearance"])
    runs: List[TrapRun] = []
    i = 0
    while i < n:
        if trace["clearance"][i] <= 0.0:
            j = i
            while j < n and trace["clearance"][j] <= 0.0:
                j += 1
            # Trapped run is [i, j)
            hazard = trace["nearest"][i]
            center = np.array([hazard[0], hazard[1]])

            speeds, headings, t_reverse = [], [], None
            for k in range(i, j):
                pos = trace["pos"][k]
                vel = trace["vel"][k]
                radial_dir = pos - center
                norm = np.linalg.norm(radial_dir)
                radial_dir = radial_dir / norm if norm > 1e-8 else radial_dir
                v_radial = float(np.dot(vel, radial_dir))
                speeds.append(float(np.linalg.norm(vel)))
                headings.append(trace["heading"][k])
                if t_reverse is None and v_radial > 0.0:
                    t_reverse = k - i

            half = max(1, len(speeds) // 2)
            rotation = sum(abs(wrap_angle(headings[k + 1] - headings[k]))
                           for k in range(len(headings) - 1))

            runs.append(TrapRun(
                dwell=j - i,
                time_to_reverse=t_reverse,
                mean_speed=float(np.mean(speeds)),
                speed_first_half=float(np.mean(speeds[:half])),
                speed_second_half=float(np.mean(speeds[half:])) if len(speeds) > half else float(np.mean(speeds[:half])),
                total_heading_rotation=rotation,
                censored=(j == n),
            ))
            i = j
        else:
            i += 1
    return runs


def summarize(runs: List[TrapRun]) -> None:
    if not runs:
        print("  (no trapped runs)")
        return
    dwells = [r.dwell for r in runs]
    reversed_runs = [r for r in runs if r.time_to_reverse is not None]
    never_reversed = len(runs) - len(reversed_runs)
    ttr = [r.time_to_reverse for r in reversed_runs]
    speed_ratio = [r.speed_second_half / r.mean_speed if r.mean_speed > 1e-6 else float("nan") for r in runs]
    rotation_rate = [r.total_heading_rotation / max(1, r.dwell) for r in runs]

    print(f"  n trapped runs: {len(runs)}  (censored/ran to episode end: {sum(r.censored for r in runs)})")
    print(f"  dwell: median={np.median(dwells):.1f} mean={np.mean(dwells):.1f} max={np.max(dwells)}")
    print(f"  time-to-reverse velocity (of runs that DID reverse, {len(reversed_runs)}/{len(runs)}): "
          f"median={np.median(ttr) if ttr else float('nan'):.1f} mean={np.mean(ttr) if ttr else float('nan'):.1f}")
    print(f"  runs that NEVER reversed velocity within the trapped window: {never_reversed}/{len(runs)}")
    print(f"  mean speed while trapped: median={np.median([r.mean_speed for r in runs]):.4f}")
    print(f"  speed(2nd half)/speed(overall) ratio: median={np.nanmedian(speed_ratio):.3f} "
          f"(< 1 means decelerating -- consistent with inertia/momentum trapping; "
          f"~1 or > 1 means speed doesn't explain the dwell)")
    print(f"  heading rotation rate while trapped (rad/step): median={np.median(rotation_rate):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--influence_radius", type=float, default=0.4)
    args = ap.parse_args()

    for robot, (env_id, checkpoint) in CHECKPOINTS.items():
        print(f"\n{'='*70}\n{robot} ({env_id})\n{'='*70}")
        shield_factory = build_shield_factory(
            env_id, shield_type="riemannian", alpha=args.alpha,
            influence_radius=args.influence_radius,
        )
        env = make_env(env_id, seed=args.seed, use_shield=True, shield_factory=shield_factory)
        model = PPO.load(checkpoint)

        all_runs: List[TrapRun] = []
        for ep in range(args.episodes):
            trace = run_episode(env, model, seed=args.seed + ep)
            all_runs.extend(extract_trap_runs(trace))
        env.close()

        summarize(all_runs)


if __name__ == "__main__":
    main()
