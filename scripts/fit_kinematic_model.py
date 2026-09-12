"""
Fit a heading-aware kinematic model for the shield's next-position prediction
and compare it against the current world-frame xy-velocity model.

Context (see SESSION_HANDOFF.md, 2026-09-11 session 4, measurement B): the
shield assumes next_pos = pos + dt * a_xy, treating the first two action
components as a world-frame xy velocity command. Measured against the real
environment dynamics, this model's prediction error is 41-53% of the hazard
radius and its direction cosine similarity with the true displacement is
statistically indistinguishable from zero, FOR BOTH SafetyPointGoal1-v0 and
SafetyCarGoal1-v0 -- a universal defect, not a car-specific one.

This script tests the natural alternative: a heading-relative (turn-and-drive)
model, fit from data rather than assumed. The model is

    disp_body = M @ a_xy + b            (fit by least squares, body frame)
    disp_world_pred = R(heading) @ disp_body

where heading is resolved via env.unwrapped.task.agent.mat (yaw =
atan2(mat[1,0], mat[0,0]), the accessor path confirmed by
scripts/probe_env_accessors.py) and R(heading) is the standard 2D rotation
matrix. Fitting M and b in the BODY frame and then rotating into world
coordinates is what "heading-relative" means operationally: the same action
should produce a fixed displacement relative to the robot's own facing
direction, regardless of which way the robot happens to be pointing in world
coordinates -- unlike the current model, which assumes the action is already
a world-frame vector independent of heading.

Does NOT wire anything into the shield. Reports numbers only.

Usage:
    .venv/bin/python -m scripts.fit_kinematic_model
    .venv/bin/python -m scripts.fit_kinematic_model --n_pairs 8000
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.envs.make_env import make_env
from src.train import build_shield_factory

HAZARD_RADIUS = 0.2  # Safety-Gymnasium default hazard radius, used throughout
# this project's other reports (e.g. SESSION_HANDOFF.md measurement B).
DT = 0.1  # matches GenericKeepoutShield / RiemannianShield's dt default,
# i.e. the current model's own assumption: next_pos = pos + dt * a_xy.

ROBOTS: Dict[str, str] = {
    "car": "SafetyCarGoal1-v0",
    "goal": "SafetyPointGoal1-v0",
}
CHECKPOINTS: Dict[str, str] = {
    "car": "checkpoints/car_riemannian_postfix/lagppo_shield_on/seed_0/latest.zip",
    "goal": "checkpoints/goal_riemannian_postfix/lagppo_shield_on/seed_0/latest.zip",
}


@dataclass
class Rollout:
    """Collected step-pair data from one robot's rollout."""

    pos_before: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pos_after: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    action_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    heading: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    intervened: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))


def get_yaw(env: Any) -> float:
    """
    Resolve the agent's current yaw from env.unwrapped.task.agent.mat.

    Accessor confirmed by scripts/probe_env_accessors.py on both
    SafetyPointGoal1-v0 and SafetyCarGoal1-v0 (3x3 rotation matrix; yaw is
    the heading of the robot's local x-axis in world coordinates).

    :param env: The (possibly wrapped) environment; env.unwrapped is used.
        :type env: Any

    :return: Yaw angle in radians.
        :rtype: float
    """
    mat = np.asarray(env.unwrapped.task.agent.mat, dtype=float).reshape(3, 3)
    return math.atan2(mat[1, 0], mat[0, 0])


def rot(theta: np.ndarray) -> np.ndarray:
    """
    Batch of 2D rotation matrices, one per angle.

    :param theta: Array of N angles, shape (N,).
        :type theta: np.ndarray

    :return: Rotation matrices, shape (N, 2, 2), such that world = R @ body.
        :rtype: np.ndarray
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((theta.shape[0], 2, 2), dtype=float)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R


def collect_rollout(env_id: str, checkpoint: str, n_pairs: int, seed: int) -> Rollout:
    """
    Roll out the actual trained checkpoint under the actual shield, collecting
    consecutive-step (pos_before, heading, action_xy, pos_after) tuples.

    Mirrors the methodology of SESSION_HANDOFF.md's measurement B: the real
    trained `*_riemannian_postfix` checkpoint, the real shield (alpha=0.1,
    influence_radius=0.4 -- the values chosen for the seeded runs), stochastic
    actions (matching how training rollouts are collected, and giving broader
    action-space coverage than a deterministic policy would for fitting
    purposes). Pairs that straddle an episode reset are dropped: pos_after
    would then belong to a different episode's initial placement, not the
    result of executing action_xy from pos_before.

    :param env_id: Safety-Gymnasium environment ID.
        :type env_id: str
    :param checkpoint: Path to the trained SB3 checkpoint.
        :type checkpoint: str
    :param n_pairs: Minimum number of valid step-pairs to collect.
        :type n_pairs: int
    :param seed: Base seed for env resets.
        :type seed: int

    :return: Collected rollout data.
        :rtype: Rollout
    """
    shield_factory = build_shield_factory(
        env_id, shield_type="riemannian", alpha=0.1, influence_radius=0.4
    )
    env = make_env(env_id, seed=seed, use_shield=True, shield_factory=shield_factory)
    model = PPO.load(checkpoint)

    pos_before: List[np.ndarray] = []
    pos_after: List[np.ndarray] = []
    action_xy: List[np.ndarray] = []
    heading: List[float] = []
    intervened: List[bool] = []

    episode = 0
    obs, _ = env.reset(seed=seed + episode)
    prev_pos = None
    prev_heading = None
    prev_action = None
    prev_intervened = None

    while len(pos_before) < n_pairs:
        yaw = get_yaw(env)
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, info = env.step(action)
        cur_pos = np.asarray(info["shield_agent_pos"], dtype=float)
        cur_action_xy = np.asarray(info["shield_safe_action_xy"], dtype=float)
        cur_intervened = bool(info.get("shield_intervened", False))

        if prev_pos is not None:
            pos_before.append(prev_pos)
            pos_after.append(cur_pos)
            action_xy.append(prev_action)
            heading.append(prev_heading)
            intervened.append(prev_intervened)

        prev_pos = cur_pos
        prev_heading = yaw
        prev_action = cur_action_xy
        prev_intervened = cur_intervened

        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            prev_pos = None  # next pair cannot cross the reset

    env.close()
    return Rollout(
        pos_before=np.array(pos_before),
        pos_after=np.array(pos_after),
        action_xy=np.array(action_xy),
        heading=np.array(heading),
        intervened=np.array(intervened, dtype=bool),
    )


def fit_body_frame_model(r: Rollout, fit_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Least-squares fit of disp_body ~= M @ a_xy + b in the robot's own body
    frame, using only the rows selected by fit_mask.

    disp_body is obtained by rotating the observed world-frame displacement
    back through the heading at the time the action was issued:
    disp_body = R(heading)^T @ (pos_after - pos_before). Fitting in the body
    frame and re-rotating by each sample's own heading at prediction time is
    what makes the resulting model heading-RELATIVE rather than a fixed
    world-frame map like the current model.

    :param r: Rollout data.
        :type r: Rollout
    :param fit_mask: Boolean mask selecting which rows to fit on.
        :type fit_mask: np.ndarray

    :return: (M, b) with M shape (2, 2), b shape (2,).
        :rtype: Tuple[np.ndarray, np.ndarray]
    """
    disp_world = r.pos_after[fit_mask] - r.pos_before[fit_mask]
    R = rot(r.heading[fit_mask])
    # world = R @ body  =>  body = R^T @ world (R is orthogonal)
    disp_body = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), disp_world)

    a = r.action_xy[fit_mask]
    X = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)  # (N, 3)
    W, _, _, _ = np.linalg.lstsq(X, disp_body, rcond=None)  # (3, 2)
    M = W[:2, :].T  # (2, 2): disp_body = M @ a + b
    b = W[2, :]
    return M, b


def predict_world_a(r: Rollout, mask: np.ndarray) -> np.ndarray:
    """Current model's prediction: pos_before + dt*a_xy, minus pos_before."""
    return DT * r.action_xy[mask]


def predict_world_b(r: Rollout, mask: np.ndarray, M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fitted heading-aware model's prediction, in world-frame displacement."""
    body_pred = r.action_xy[mask] @ M.T + b
    R = rot(r.heading[mask])
    return np.einsum("nij,nj->ni", R, body_pred)


def fit_world_frame_control(r: Rollout, fit_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Control model C: a fixed WORLD-frame affine map disp_world ~= C @ a_xy + c,
    fit the same way as model B but with no heading rotation at all.

    This isolates whether heading is doing real work in model B, versus model
    B's improvement being explainable by simply recalibrating a constant
    linear map (scale + axis coupling) in world coordinates -- something a
    fixed robot orientation habit could produce without any heading-relative
    mechanism being present.

    :param r: Rollout data.
        :type r: Rollout
    :param fit_mask: Boolean mask selecting which rows to fit on.
        :type fit_mask: np.ndarray

    :return: (C, c) with C shape (2, 2), c shape (2,).
        :rtype: Tuple[np.ndarray, np.ndarray]
    """
    disp_world = r.pos_after[fit_mask] - r.pos_before[fit_mask]
    a = r.action_xy[fit_mask]
    X = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    W, _, _, _ = np.linalg.lstsq(X, disp_world, rcond=None)
    return W[:2, :].T, W[2, :]


def predict_world_c(r: Rollout, mask: np.ndarray, C: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Control model's prediction: a fixed world-frame affine map, no heading."""
    return r.action_xy[mask] @ C.T + c


def summarize(name: str, pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """
    Error norm and direction cosine summary for one (pred, actual) pair set.

    :param name: Label, unused except for readability at call sites.
        :type name: str
    :param pred: Predicted world-frame displacements, shape (N, 2).
        :type pred: np.ndarray
    :param actual: Actual world-frame displacements, shape (N, 2).
        :type actual: np.ndarray

    :return: Dict of summary statistics.
        :rtype: Dict[str, float]
    """
    err = np.linalg.norm(pred - actual, axis=1)
    pred_n = np.linalg.norm(pred, axis=1)
    act_n = np.linalg.norm(actual, axis=1)
    valid = (pred_n > 1e-6) & (act_n > 1e-6)
    cos = np.sum(pred[valid] * actual[valid], axis=1) / (pred_n[valid] * act_n[valid])
    return {
        "n": len(err),
        "mean_err": float(np.mean(err)),
        "median_err": float(np.median(err)),
        "median_err_frac_radius": float(np.median(err) / HAZARD_RADIUS),
        "n_cos": int(valid.sum()),
        "mean_cos": float(np.mean(cos)) if valid.any() else float("nan"),
        "median_cos": float(np.median(cos)) if valid.any() else float("nan"),
        "frac_cos_gt_0.9": float(np.mean(cos > 0.9)) if valid.any() else float("nan"),
        "frac_cos_lt_0": float(np.mean(cos < 0.0)) if valid.any() else float("nan"),
    }


def print_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    """Print a compact fixed-width table of summarize() results."""
    header = (
        f"{'row':<28} {'n':>6} {'mean_err':>9} {'med_err':>8} {'med/r':>7} "
        f"{'mean_cos':>9} {'med_cos':>8} {'>0.9':>6} {'<0':>6}"
    )
    print(header)
    print("-" * len(header))
    for label, s in rows:
        print(
            f"{label:<28} {s['n']:>6} {s['mean_err']:>9.4f} {s['median_err']:>8.4f} "
            f"{s['median_err_frac_radius']:>7.3f} {s['mean_cos']:>9.4f} "
            f"{s['median_cos']:>8.4f} {s['frac_cos_gt_0.9']:>6.3f} {s['frac_cos_lt_0']:>6.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=6000,
                     help="Minimum valid step-pairs to collect per robot.")
    ap.add_argument("--seed", type=int, default=100,
                     help="Base seed (matches the shielded-eval protocol's seed=100).")
    ap.add_argument("--fit_frac", type=float, default=0.5,
                     help="Fraction of non-intervened pairs used to FIT model B; "
                          "the rest are held out for evaluation, same as model A.")
    args = ap.parse_args()

    rng = np.random.default_rng(0)

    for robot, env_id in ROBOTS.items():
        print(f"\n{'='*70}\n{robot} ({env_id})\n{'='*70}")
        r = collect_rollout(env_id, CHECKPOINTS[robot], args.n_pairs, args.seed)
        print(f"collected {len(r.pos_before)} valid step-pairs "
              f"({int(r.intervened.sum())} intervened, "
              f"{int((~r.intervened).sum())} non-intervened)")

        non_interv = np.where(~r.intervened)[0]
        rng.shuffle(non_interv)
        split = int(len(non_interv) * args.fit_frac)
        fit_idx = non_interv[:split]
        holdout_non_interv_idx = non_interv[split:]

        fit_mask = np.zeros(len(r.pos_before), dtype=bool)
        fit_mask[fit_idx] = True

        M, b = fit_body_frame_model(r, fit_mask)
        print(f"fitted body-frame model: M=\n{M}\nb={b}")
        C, c = fit_world_frame_control(r, fit_mask)
        print(f"control world-frame-only model: C=\n{C}\nc={c}")

        holdout_mask = np.zeros(len(r.pos_before), dtype=bool)
        holdout_mask[holdout_non_interv_idx] = True
        interv_mask = r.intervened
        all_holdout_mask = holdout_mask | interv_mask  # never includes fit rows

        actual_holdout_non = r.pos_after[holdout_mask] - r.pos_before[holdout_mask]
        actual_interv = r.pos_after[interv_mask] - r.pos_before[interv_mask]
        actual_all_holdout = r.pos_after[all_holdout_mask] - r.pos_before[all_holdout_mask]

        rows = []
        rows.append((
            "A world-frame, non-interv(holdout)",
            summarize("A", predict_world_a(r, holdout_mask), actual_holdout_non),
        ))
        rows.append((
            "B fitted heading, non-interv(holdout)",
            summarize("B", predict_world_b(r, holdout_mask, M, b), actual_holdout_non),
        ))
        rows.append((
            "A world-frame, intervened",
            summarize("A", predict_world_a(r, interv_mask), actual_interv),
        ))
        rows.append((
            "B fitted heading, intervened",
            summarize("B", predict_world_b(r, interv_mask, M, b), actual_interv),
        ))
        rows.append((
            "A world-frame, ALL holdout",
            summarize("A", predict_world_a(r, all_holdout_mask), actual_all_holdout),
        ))
        rows.append((
            "B fitted heading, ALL holdout",
            summarize("B", predict_world_b(r, all_holdout_mask, M, b), actual_all_holdout),
        ))
        rows.append((
            "C world-frame-only ctrl, non-interv",
            summarize("C", predict_world_c(r, holdout_mask, C, c), actual_holdout_non),
        ))
        rows.append((
            "C world-frame-only ctrl, ALL holdout",
            summarize("C", predict_world_c(r, all_holdout_mask, C, c), actual_all_holdout),
        ))
        print_table(rows)


if __name__ == "__main__":
    main()
