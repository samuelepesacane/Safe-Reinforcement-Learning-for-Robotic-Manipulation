"""
Probe Safety-Gymnasium for agent position and heading accessors.

This unblocks two things at once:

  Run 0 (D1) -- the shield currently reads the agent's XY from obs[:2], but
  Safety-Gymnasium observations are proprioceptive sensors plus pseudo-lidar and
  contain no absolute position at all. obs[:2] is the first two accelerometer
  components. The shield needs a real position source from env.unwrapped.

  Run 3 -- the action wrapper that cripples lateral motion needs the robot's
  current heading, which is likewise absent from the 60-dim PointGoal observation.

The audit could not resolve these accessor names because safety_gymnasium is not
installed on the Windows machine where it ran. This script resolves them on the
WSL box empirically rather than by guessing, and asserts that whatever it finds
actually tracks the robot instead of merely existing.

Usage:
    python scripts/probe_env_accessors.py
    python scripts/probe_env_accessors.py --env_id SafetyCarGoal1-v0

Read the PASS/FAIL block at the end; that is the whole output that matters.
"""

import argparse
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def try_paths(env: Any, paths: List[str]) -> Dict[str, Any]:
    """
    Evaluate a list of dotted attribute paths against an env, keeping the hits.

    Attribute access on a partially-initialized MuJoCo wrapper can raise almost
    anything, so every probe is individually guarded and failures are simply
    absent from the result.

    :param env: The environment to probe (usually env.unwrapped).
        :type env: Any
    :param paths: Dotted attribute paths, e.g. "task.agent.pos".
        :type paths: List[str]

    :return: Mapping from path to the value found there.
        :rtype: Dict[str, Any]
    """
    found: Dict[str, Any] = {}
    for path in paths:
        node: Any = env
        try:
            for part in path.split("."):
                node = getattr(node, part)
            if callable(node):
                node = node()
            found[path] = node
        except Exception:
            continue
    return found


def summarize(value: Any) -> str:
    """
    Render a probed value compactly enough for a terminal table.

    :param value: Whatever the attribute path yielded.
        :type value: Any

    :return: One-line description.
        :rtype: str
    """
    try:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return f"scalar {arr:.4f}"
        return f"shape {arr.shape} {np.array2string(arr.ravel()[:9], precision=4)}"
    except Exception:
        return f"{type(value).__name__}"


POSITION_PATHS = [
    "task.agent.pos",
    "task.agent.body_pos",
    "agent.pos",
    "task.robot.pos",
    "robot_pos",
    "task.agent_pos",
    "world.robot_pos",
]

ROTATION_PATHS = [
    "task.agent.mat",
    "task.agent.rot",
    "task.agent.quat",
    "task.agent.body_quat",
    "agent.mat",
    "task.robot.mat",
]

HAZARD_PATHS = [
    "task.hazards.pos",
    "task.hazards_pos",
    "world.hazards_pos",
    "task.hazards.size",
    "task.hazards_size",
]


def main() -> None:
    """
    Probe one environment and report which accessors are usable.

    Three checks are run, and all three must pass before Run 0 or Run 3 can be
    implemented against the discovered names:

      1. A position accessor exists and MOVES when the robot is driven. An
         accessor that exists but returns a constant is worse than none, because
         it would silently reproduce the D1 failure in a new place.
      2. A rotation accessor exists and yields a usable yaw.
      3. Hazard positions exist AND change across episode resets, which is the
         premise behind D2.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="SafetyPointGoal1-v0")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    import safety_gymnasium  # noqa: F401
    import gymnasium as gym

    env = gym.make(args.env_id, disable_env_checker=True)
    obs, _ = env.reset(seed=0)
    uw = env.unwrapped

    obs_arr = np.asarray(obs, dtype=float)
    print(f"env      : {args.env_id}")
    print(f"obs      : shape {obs_arr.shape}")
    print(f"obs[:4]  : {np.array2string(obs_arr[:4], precision=4)}")
    print(f"action   : {env.action_space}")
    print()

    print("--- position candidates " + "-" * 46)
    pos_hits = try_paths(uw, POSITION_PATHS)
    for k, v in pos_hits.items():
        print(f"  {k:<28} {summarize(v)}")
    if not pos_hits:
        print("  (none)")

    print("\n--- rotation candidates " + "-" * 46)
    rot_hits = try_paths(uw, ROTATION_PATHS)
    for k, v in rot_hits.items():
        print(f"  {k:<28} {summarize(v)}")
    if not rot_hits:
        print("  (none)")

    print("\n--- hazard candidates " + "-" * 48)
    haz_hits = try_paths(uw, HAZARD_PATHS)
    for k, v in haz_hits.items():
        print(f"  {k:<28} {summarize(v)}")
    if not haz_hits:
        print("  (none)")

    # Check 1: does a position accessor actually track the robot?
    print("\n--- check 1: position tracks motion " + "-" * 34)
    moving: List[Tuple[str, float]] = []
    for path in pos_hits:
        before = np.asarray(try_paths(uw, [path])[path], dtype=float).ravel()[:2].copy()
        for _ in range(args.steps):
            env.step(np.ones(env.action_space.shape, dtype=np.float32))
        after = np.asarray(try_paths(uw, [path])[path], dtype=float).ravel()[:2]
        travelled = float(np.linalg.norm(after - before))
        status = "MOVES" if travelled > 1e-3 else "static"
        print(f"  {path:<28} |delta| = {travelled:.5f}  {status}")
        if travelled > 1e-3:
            moving.append((path, travelled))
        env.reset(seed=0)

    # Also confirm the D1 diagnosis directly: obs[:2] must NOT track position.
    obs2, _ = env.reset(seed=0)
    obs_before = np.asarray(obs2, dtype=float)[:2].copy()
    for _ in range(args.steps):
        obs2, *_ = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    obs_after = np.asarray(obs2, dtype=float)[:2]
    if moving:
        ref_path = moving[0][0]
        ref = np.asarray(try_paths(uw, [ref_path])[ref_path], dtype=float).ravel()[:2]
        print(f"\n  obs[:2] moved {np.linalg.norm(obs_after - obs_before):.5f}; "
              f"true position is {np.array2string(ref, precision=4)}, "
              f"obs[:2] is {np.array2string(obs_after, precision=4)}")
        print("  ^ if these two differ, D1 is confirmed empirically")

    # Check 2: yaw
    print("\n--- check 2: heading " + "-" * 49)
    yaw: Optional[float] = None
    for path, val in rot_hits.items():
        arr = np.asarray(val, dtype=float)
        try:
            if arr.size == 9:
                m = arr.reshape(3, 3)
                yaw = math.atan2(m[1, 0], m[0, 0])
            elif arr.size == 4:
                w, x, y, z = arr.ravel()
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            if yaw is not None:
                print(f"  {path:<28} yaw = {yaw:+.4f} rad ({math.degrees(yaw):+.1f} deg)")
        except Exception as exc:
            print(f"  {path:<28} unusable: {exc}")

    # Check 3: hazards re-randomize across resets
    print("\n--- check 3: hazards change across resets " + "-" * 28)
    haz_path = next((p for p in ("task.hazards.pos", "task.hazards_pos", "world.hazards_pos")
                     if p in haz_hits), None)
    hazards_change = False
    if haz_path:
        env.reset(seed=0)
        first = np.asarray(try_paths(uw, [haz_path])[haz_path], dtype=float).copy()
        env.reset(seed=1)
        second = np.asarray(try_paths(uw, [haz_path])[haz_path], dtype=float)
        hazards_change = not np.allclose(first, second)
        print(f"  {haz_path}: layout {'CHANGES' if hazards_change else 'is identical'} "
              f"across resets")
        print(f"    reset(0) first hazard: {np.array2string(first.ravel()[:3], precision=4)}")
        print(f"    reset(1) first hazard: {np.array2string(second.ravel()[:3], precision=4)}")
    else:
        print("  no hazard accessor found")

    env.close()

    print("\n" + "=" * 70)
    ok_pos = bool(moving)
    ok_yaw = yaw is not None
    ok_haz = hazards_change
    print(f"  position accessor : {'PASS  -> ' + moving[0][0] if ok_pos else 'FAIL'}")
    print(f"  heading accessor  : {'PASS' if ok_yaw else 'FAIL'}")
    print(f"  hazards re-randomize (confirms D2): {'PASS' if ok_haz else 'FAIL'}")
    print("=" * 70)
    if not (ok_pos and ok_yaw):
        print("\nRun 0 and Run 3 are BLOCKED until position and heading both PASS.")
        print("Paste this whole output back and the candidate path lists will be widened.")
