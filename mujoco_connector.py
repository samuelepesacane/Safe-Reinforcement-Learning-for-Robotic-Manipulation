"""
Gymnasium wrapper for loading any MuJoCo .xml model into the Safe RL pipeline, 
bypassing the Safety-Gymnasium abstraction layer.

MujocoRoboticEnv wraps a raw .xml file and exposes the standard Gymnasium interface 
(observations, actions, reward, cost, and an info dict) with exactly the info keys 
that CostInfoWrapper, LagrangianCallback, ShieldingActionWrapper, and evaluate.py expect.

MujocoShield is a pass-through shield satisfying the interface expected by
ShieldingActionWrapper. It does no geometry-based projection; replace the
body of step() with a CBF or QP solver for a real safety filter.

Integration requires two small edits to the existing pipeline:

1. In src/envs/make_env.py, patch _try_make() to intercept env_id strings
   prefixed with "mujoco:" before they reach gym.make(), which would crash
   on an unregistered prefix.
 
2. In src/train.py, add a guard at the top of build_shield_factory() so it
   does not try to read env.unwrapped.world.hazards_pos, a Safety-Gymnasium
   attribute that does not exist on a raw MuJoCo env.

Note: MujocoRoboticEnv cannot be pickled across subprocesses, so
SubprocVecEnv will crash when num_envs > 1. Always pass --num_envs 1
when training with this connector.

Tested on mujoco 2.3.0. The mujoco.viewer module was added in 2.3.7.,
on older installs (such as 2.3.0), render_mode="human" is silently skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False
    print("[mujoco_connector] WARNING: mujoco not found.  pip install mujoco")

try:
    import mujoco.viewer
    _HAS_VIEWER = True
except ImportError:
    # mujoco.viewer was added in 2.3.7; on older installs we run headless only
    _HAS_VIEWER = False


class MujocoRoboticEnv(gym.Env):
    """
    Gymnasium environment wrapping any MuJoCo .xml model.
 
    Loads the model from a file path, builds observations from joint states
    and optional site positions, computes a distance-based reward, and reports
    a safety cost as penetration depth into user-defined hazard geometries.
    The info dict uses exactly the keys that the rest of the pipeline expects,
    so CostInfoWrapper, ShieldingActionWrapper, and RewardShapingWrapper are
    applied on top by make_env() without any special casing here.
 
    Observation: qpos || qvel [|| end-effector xyz || goal xyz]
    Action: normalised joint torques in [-1, 1], rescaled to each actuator's
    physical ctrlrange at step time.
    Reward: negative distance from end-effector to goal, plus a success bonus
    when the goal threshold is reached, minus a control cost.
    Cost: maximum penetration depth into any hazard geom, clipped to [0, 1].
 
    :param model_path: Path to the MuJoCo .xml file.
        :type model_path: str
    :param render_mode: Rendering mode, one of "human" or "rgb_array".
        :type render_mode: str, optional
    :param goal_site: Name of the goal site in the .xml model.
        :type goal_site: str
    :param ee_site: Name of the end-effector site in the .xml model.
        :type ee_site: str
    :param goal_threshold: Distance below which the task is considered solved.
        :type goal_threshold: float
    :param hazard_geoms: List of geom names to treat as hazard zones.
        :type hazard_geoms: List[str], optional
    :param safety_radius: Radius around each hazard geom that defines the unsafe zone.
        :type safety_radius: float
    :param ctrl_cost_weight: Weight applied to the squared control norm penalty.
        :type ctrl_cost_weight: float
    :param max_episode_steps: Maximum steps before the episode is truncated.
        :type max_episode_steps: int
    :param frame_skip: Number of MuJoCo simulation steps per environment step.
        :type frame_skip: int
    """
 
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
 
    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        goal_site: str = "target",
        ee_site: str = "end_effector",
        goal_threshold: float = 0.05,
        hazard_geoms: Optional[List[str]] = None,
        safety_radius: float = 0.15,
        ctrl_cost_weight: float = 0.001,
        max_episode_steps: int = 500,
        frame_skip: int = 5,
    ):
        if not _HAS_MUJOCO:
            raise RuntimeError("pip install mujoco")
        super().__init__()
 
        self.render_mode       = render_mode
        self.goal_site         = goal_site
        self.ee_site           = ee_site
        self.goal_threshold    = goal_threshold
        self.hazard_geoms      = hazard_geoms or []
        self.safety_radius     = safety_radius
        self.ctrl_cost_weight  = ctrl_cost_weight
        self.max_episode_steps = max_episode_steps
        self.frame_skip        = frame_skip
 
        # Resolve to an absolute path so the env works regardless of the
        # working directory from which it is constructed
        path = str(Path(model_path).expanduser().resolve())
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data  = mujoco.MjData(self.model)
        self._viewer     = None
        self._step_count = 0
 
        # Include end-effector and goal positions in the observation only if
        # those sites exist in the model, so the connector works with any .xml
        self._has_ee   = self._site_exists(ee_site)
        self._has_goal = self._site_exists(goal_site)
        obs_dim = self.model.nq + self.model.nv
        obs_dim += 3 * int(self._has_ee) + 3 * int(self._has_goal)
        hi = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
 
        # Action space is always [-1, 1]; rescaling to physical limits happens at step time
        nu = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)
 
        # Precompute affine rescaling coefficients so the step loop is cheap
        cr = self.model.actuator_ctrlrange          # shape (nu, 2)
        self._act_mid = 0.5 * (cr[:, 1] + cr[:, 0])
        self._act_rng = 0.5 * (cr[:, 1] - cr[:, 0])
 
        # Resolve hazard geom names to IDs once at construction to avoid
        # repeated name lookups inside the step loop
        self._hazard_ids: List[int] = []
        for name in self.hazard_geoms:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._hazard_ids.append(gid)
            else:
                print(f"[mujoco_connector] WARNING: geom '{name}' not found in model")
 
    def _site_exists(self, name: str) -> bool:
        """
        Check whether a named site exists in the loaded model.
 
        MuJoCo returns -1 from mj_name2id when the name is not found, so we
        use that as our existence test rather than inspecting the model XML.
 
        :param name: Site name to look up.
            :type name: str
 
        :return: True if the site exists in the model.
            :rtype: bool
        """
        return mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
 
    def _site_pos(self, name: str) -> np.ndarray:
        """
        Return the current world-frame position of a named site.
 
        Site positions in data.site_xpos are updated by mj_forward and
        mj_step, so this always reflects the current simulation state.
 
        :param name: Site name to look up.
            :type name: str
 
        :return: World-frame position, shape (3,).
            :rtype: np.ndarray
        """
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid].copy()
 
    def _obs(self) -> np.ndarray:
        """
        Build the observation vector from the current simulation state.
 
        Joint positions and velocities are always included. End-effector and
        goal site positions are appended when those sites exist, giving the
        policy direct access to task geometry without having to infer it from
        joint angles alone.
 
        :return: Observation vector, shape (obs_dim,), dtype float32.
            :rtype: np.ndarray
        """
        parts = [self.data.qpos.copy(), self.data.qvel.copy()]
        if self._has_ee:
            parts.append(self._site_pos(self.ee_site))
        if self._has_goal:
            parts.append(self._site_pos(self.goal_site))
        return np.concatenate(parts).astype(np.float32)
 
    def _reward(self) -> float:
        """
        Compute the per-step reward from the current simulation state.
 
        Negative distance from end-effector to goal gives a dense learning
        signal at every step. A discrete bonus is added when the goal threshold
        is reached, and a small control cost penalises large torques to
        encourage smooth trajectories.
 
        Returns 0 if either site is missing so the simulation runs without a
        spatial objective on models that lack these sites.
 
        :return: Scalar reward for the current step.
            :rtype: float
        """
        if not (self._has_ee and self._has_goal):
            return 0.0
        dist = float(np.linalg.norm(
            self._site_pos(self.ee_site) - self._site_pos(self.goal_site)))
        r = -dist + (1.0 if dist < self.goal_threshold else 0.0)
        r -= self.ctrl_cost_weight * float(np.sum(self.data.ctrl ** 2))
        return r
 
    def _cost(self) -> float:
        """
        Compute the per-step safety cost from the current simulation state.
 
        The cost is the maximum penetration depth of the end-effector into any
        hazard geom, normalised to [0, 1]: 0 means fully outside all hazard
        zones, 1 means at the centre of the closest hazard.
 
        Returns 0 if the end-effector site is missing or no hazard geoms were
        specified. In that case the Lagrangian multiplier stays at zero, which
        is correct when no constraint is ever violated.
 
        :return: Safety cost in [0, 1].
            :rtype: float
        """
        if not self._has_ee or not self._hazard_ids:
            return 0.0
        ee   = self._site_pos(self.ee_site)
        cost = 0.0
        for gid in self._hazard_ids:
            d = float(np.linalg.norm(ee - self.data.geom_xpos[gid]))
            if d < self.safety_radius:
                # Penetration depth: 0 at the boundary, 1 at the centre
                cost = max(cost, 1.0 - d / self.safety_radius)
        return cost
 
    def reset(self, *, seed=None, options=None):
        """
        Reset the simulation to a slightly randomised initial configuration.
 
        A small uniform perturbation is added to the default joint positions so
        the policy sees varied starting states rather than always beginning from
        the same configuration, which would reduce generalisation.
 
        :param seed: Random seed passed to the parent Gymnasium Env.
            :type seed: int, optional
        :param options: Unused; kept for API compatibility.
            :type options: dict, optional
 
        :return: Tuple of (observation, info dict).
            :rtype: tuple[np.ndarray, dict]
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Small jitter prevents the policy from memorising a single fixed trajectory
        self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, self.model.nq)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._obs(), {}
 
    def step(self, action: np.ndarray):
        """
        Advance the simulation by one environment step.
 
        The normalised action is rescaled to physical actuator limits before
        being applied. frame_skip sub-steps are simulated per environment step
        so that the effective control frequency is lower than the physics
        frequency, which is standard practice in MuJoCo environments.
 
        The info dict uses exactly the keys expected by the pipeline:
        CostInfoWrapper reads "cost", metrics.py reads "cost_sum", and
        evaluate.py reads "is_success". ShieldingActionWrapper overwrites
        "shield_intervened" with the true intervention flag after this returns.
 
        :param action: Normalised joint torques, shape (nu,), values in [-1, 1].
            :type action: np.ndarray
 
        :return: Tuple of (obs, reward, terminated, truncated, info).
            :rtype: tuple[np.ndarray, float, bool, bool, dict]
        """
        # Rescale from [-1, 1] to each actuator's physical control range
        self.data.ctrl[:] = (
            self._act_mid + np.clip(action, -1., 1.) * self._act_rng)
 
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
 
        obs  = self._obs()
        rew  = self._reward()
        cost = self._cost()
        self._step_count += 1
 
        trunc = self._step_count >= self.max_episode_steps
        term  = bool(
            self._has_ee and self._has_goal and
            float(np.linalg.norm(
                self._site_pos(self.ee_site) -
                self._site_pos(self.goal_site)
            )) < self.goal_threshold
        )
        info: Dict[str, Any] = {
            "cost":              cost,
            "cost_sum":          cost,   # alias expected by metrics.py
            "is_success":        term,
            "shield_intervened": False,  # ShieldingActionWrapper overwrites this
        }
        if self.render_mode == "human":
            self.render()
        return obs, rew, term, trunc, info
 
    def render(self):
        """
        Render the current simulation state.
 
        In "human" mode, a passive viewer window is opened on the first call
        and kept alive across steps. In "rgb_array" mode, a renderer is
        created per call and the image array is returned.
 
        On mujoco < 2.3.7, "human" mode is silently skipped because
        mujoco.viewer does not exist.
 
        :return: RGB image array of shape (H, W, 3) in "rgb_array" mode, else None.
            :rtype: np.ndarray or None
        """
        if self.render_mode == "human":
            if _HAS_VIEWER:
                if self._viewer is None:
                    self._viewer = mujoco.viewer.launch_passive(
                        self.model, self.data)
                self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model)
            renderer.update_scene(self.data)
            return renderer.render()
 
    def close(self):
        """
        Close the viewer window if one is open.
 
        Failing to close the viewer can leave orphan processes on some platforms.
 
        :return: None.
            :rtype: None
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
 
 
class MujocoShield:
    """
    Pass-through shield satisfying the interface expected by ShieldingActionWrapper.
 
    ShieldingActionWrapper calls shield.step(action, obs) at each step, reads
    shield.last_intervened to decide whether to log an intervention, and calls
    shield.on_reset() at episode boundaries.
 
    This implementation is a no-op: it returns the action unchanged and always
    reports last_intervened=False. To add real geometric safety projection,
    replace the body of step() with a CBF or QP-based filter.
 
    :param last_intervened: Whether the shield modified the last action.
        :type last_intervened: bool
    """
 
    def __init__(self):
        self.last_intervened: bool = False
 
    def step(self, action: np.ndarray, obs: Any) -> np.ndarray:
        """
        Return the action unchanged.
 
        In a real shield this would check whether the action would violate a
        safety constraint and project it to the nearest safe action if so.
 
        :param action: Proposed action from the policy, shape (nu,).
            :type action: np.ndarray
        :param obs: Current observation. Unused; included for interface compatibility.
            :type obs: Any
 
        :return: Action unchanged, same shape as input.
            :rtype: np.ndarray
        """
        self.last_intervened = False
        return action
 
    def on_reset(self) -> None:
        """
        Reset internal shield state at the start of a new episode.
 
        Clears last_intervened so intervention counts do not bleed across episodes.
 
        :return: None.
            :rtype: None
        """
        self.last_intervened = False
 
 
def _demo(model_path: str, render: bool, n_steps: int) -> None:
    """
    Run a short random-policy episode and print basic diagnostics.
 
    Sanity check that the connector loads the model, builds observation and
    action spaces correctly, and that reward and cost signals are non-trivial.
    Does not test full pipeline integration.
 
    :param model_path: Path to the MuJoCo .xml file.
        :type model_path: str
    :param render: If True, open a viewer window during the episode.
        :type render: bool
    :param n_steps: Number of environment steps to run.
        :type n_steps: int
 
    :return: None.
        :rtype: None
    """
    print(f"\n=== mujoco_connector demo  model={model_path} ===\n")
    env = MujocoRoboticEnv(
        model_path  = model_path,
        render_mode = "human" if render else None,
    )
    obs, _ = env.reset(seed=0)
    print(f"obs shape : {obs.shape}")
    print(f"act space : {env.action_space}")
    print(f"ee site found   : {env._has_ee}")
    print(f"goal site found : {env._has_goal}\n")
 
    total_r = total_c = 0.0
    t0 = time.time()
    for s in range(n_steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        total_c += info["cost"]
        if (s + 1) % 50 == 0:
            print(f"  step {s+1:4d}  r={r:+.4f}  cost={info['cost']:.4f}  "
                  f"success={info['is_success']}")
        if term or trunc:
            print(f"  -- episode ended at step {s+1} (term={term}) --")
            env.reset()
 
    fps = n_steps / (time.time() - t0)
    print(f"\ntotal_reward={total_r:.3f}  total_cost={total_c:.3f}  "
          f"fps={fps:.1f}")
    env.close()
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MuJoCo Safe-RL connector demo")
    ap.add_argument("--model_path",  default="assets/robot.xml")
    ap.add_argument("--render",      action="store_true")
    ap.add_argument("--steps",       type=int, default=200)
    args = ap.parse_args()
    _demo(args.model_path, args.render, args.steps)
 