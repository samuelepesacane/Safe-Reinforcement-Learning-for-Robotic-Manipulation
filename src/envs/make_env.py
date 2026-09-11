from typing import Callable, List, Optional, Any, Dict, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from mujoco_connector import MujocoRoboticEnv
from stable_baselines3.common.monitor import Monitor

# Try to import Safety-Gymnasium so that env IDs like "SafetyPointPush1-v0"
# are registered with Gymnasium. If it's not installed, gym.make() will fail
# later with an informative error, so we only warn here rather than raising.
try:
    import safety_gymnasium  # noqa: F401
except Exception:
    print("WARNING: safety_gymnasium not imported")
    pass


class NormalizeStepReturn(gym.Wrapper):
    """
    Normalize env.step output to the Gymnasium 5-tuple:
    (obs, reward, terminated, truncated, info).

    Handles four common layouts:
    - 4-tuple (old Gym API): (obs, reward, done, info)
    - 5-tuple (Gymnasium): (obs, reward, terminated, truncated, info)
    - 6-tuple safety format: (obs, reward, cost, terminated, truncated, info)
    - 6-tuple extra payload: (obs, reward, terminated, truncated, info, extra)

    In the 6-tuple safety case, the scalar cost is moved into info["cost"].
    In the extra-payload case, the extra fields are merged into info.
    """

    def step(self, action):
        out = self.env.step(action)
        if not isinstance(out, tuple):
            raise RuntimeError(f"env.step returned non-tuple: {type(out)}")

        n = len(out)

        # Old Gym API: (obs, reward, done, info)
        if n == 4:
            obs, reward, done, info = out
            info = dict(info) if info is not None else {}
            return obs, reward, bool(done), False, info

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        if n == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            return obs, reward, bool(terminated), bool(truncated), info

        # 6-element layouts: safety cost or extra payload
        if n == 6:
            third, last = out[2], out[5]

            # Layout A: (obs, reward, cost, terminated, truncated, info)
            if isinstance(third, (int, float, np.integer, np.floating)) and (
                isinstance(last, dict) or last is None
            ):
                obs, reward, cost, terminated, truncated, info = out
                info = dict(info) if info is not None else {}
                info.setdefault("cost", float(cost))
                return obs, reward, bool(terminated), bool(truncated), info

            # Layout B: (obs, reward, terminated, truncated, info, extra)
            obs, reward, terminated, truncated, info, info_extra = out
            info = dict(info) if info is not None else {}
            if isinstance(info_extra, dict):
                # Merge extra keys; prefix on collision to avoid silently dropping data
                for k, v in info_extra.items():
                    key = f"extra_{k}" if k in info else k
                    info[key] = v
            else:
                info["extra"] = info_extra
            return obs, reward, bool(terminated), bool(truncated), info

        raise RuntimeError(f"Unexpected env.step tuple length: {n}")


class CostInfoWrapper(gym.Wrapper):
    """
    Guarantee a scalar cost in info at every step and track episodic cost.

    Safety-Gymnasium envs expose cost natively; plain Gymnasium envs do not.
    This wrapper normalizes both cases so downstream code never needs to
    check which env type it's talking to.

    When an episode terminates or is truncated, the wrapper adds:
        info["episodic_cost"]
        info["episodic_length"]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episodic_cost = 0.0
        self.steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episodic_cost = 0.0
        self.steps = 0
        return obs, info

    def step_with_cost(self, action):
        """
        Normalize the env output and ensure info["cost"] exists.

        Accepts both the 6-tuple Safety-Gymnasium layout and the 5-tuple
        Gymnasium layout. Returns a standard 5-tuple in either case.

        :param action: Action to pass to the wrapped environment.
            :type action: Any

        :return: Standard 5-tuple (obs, reward, terminated, truncated, info)
            with info["cost"] always present.
            :rtype: tuple
        """
        out = self.env.step(action)
        if not isinstance(out, tuple):
            raise RuntimeError(f"env.step returned non-tuple: {out!r}")

        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            info["cost"] = float(cost)
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            info.setdefault("cost", float(info.get("cost", 0.0)))
        else:
            raise RuntimeError(f"Unexpected env.step tuple length: {len(out)}")

        return obs, reward, terminated, truncated, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.step_with_cost(action)

        cost = float(info.get("cost", 0.0))
        self.episodic_cost += cost
        self.steps += 1

        # Attach episodic summaries at episode end so the training logger
        # can read them without needing access to this wrapper directly
        if terminated or truncated:
            info["episodic_cost"] = self.episodic_cost
            info["episodic_length"] = self.steps

        return obs, rew, terminated, truncated, info


class RewardShapingWrapper(gym.RewardWrapper):
    """
    Lagrangian reward shaping for constrained RL:

        r_shaped = reward - lambda * cost

    The current lambda value is provided by the callable get_lambda, which
    allows online adaptation (e.g. dual gradient updates in LagPPO and RCPO).
    The wrapper logs both shaped_reward and lambda in the info dict so the
    training logger can track the multiplier trajectory.
    """

    def __init__(self, env: gym.Env, get_lambda: Callable[[], float]):
        super().__init__(env)
        self.get_lambda = get_lambda

    def step_with_cost(self, action):
        """
        Normalize the env output and ensure info["cost"] exists.

        Mirrors the logic in CostInfoWrapper. Kept local here to avoid
        coupling the wrapper order to the wrapping logic.

        :param action: Action to pass to the wrapped environment.
            :type action: Any

        :return: Standard 5-tuple (obs, reward, terminated, truncated, info)
            with info["cost"] always present.
            :rtype: tuple
        """
        out = self.env.step(action)
        if not isinstance(out, tuple):
            raise RuntimeError(f"env.step returned non-tuple: {out!r}")

        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            info["cost"] = float(cost)
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            info.setdefault("cost", float(info.get("cost", 0.0)))
        else:
            raise RuntimeError(f"Unexpected env.step tuple length: {len(out)}")

        return obs, reward, terminated, truncated, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.step_with_cost(action)

        cost = float(info.get("cost", 0.0))
        lam = float(self.get_lambda() if self.get_lambda is not None else 0.0)
        shaped = float(reward) - lam * cost

        info["shaped_reward"] = shaped
        info["lambda"] = lam
        return obs, shaped, terminated, truncated, info


def resolve_agent_pos(env: gym.Env) -> np.ndarray:
    """
    Resolve the agent's true 2D position from the environment.

    Safety-Gymnasium's flat observation vector (sensors + pseudo-lidar) has no
    absolute position in it -- obs[:2] there is the first two accelerometer
    components, not a position (this was D1). The real position lives on
    env.unwrapped, verified empirically against SafetyPointGoal1-v0 and
    SafetyCarGoal1-v0 via scripts/probe_env_accessors.py: the primary path is
    the same env.unwrapped.task.agent object whose .pos attribute the probe
    confirmed tracks the robot (not merely exists) on both robot types.

    Raises rather than returning None on failure, so a broken accessor fails
    loudly instead of silently degrading the shield to a pass-through.

    :param env: The (possibly wrapped) environment; env.unwrapped is used.
        :type env: gym.Env

    :return: 2D position [x, y].
        :rtype: np.ndarray

    :raises RuntimeError: If no known accessor path yields a position.
    """
    uw = env.unwrapped

    try:
        if hasattr(uw, "task") and hasattr(uw.task, "agent"):
            pos = getattr(uw.task.agent, "pos", None)
            if pos is not None:
                return np.array(pos[:2], dtype=np.float32)
    except Exception:
        pass

    raise RuntimeError(
        "Could not resolve agent position from env.unwrapped: expected "
        "env.unwrapped.task.agent.pos (verified via scripts/probe_env_accessors.py "
        "on SafetyPointGoal1-v0 and SafetyCarGoal1-v0). If this fires on a "
        "different env, widen the probe's POSITION_PATHS and this cascade "
        "together -- do not silently pass the action through unshielded."
    )


def resolve_hazards(env: gym.Env) -> Optional[List[Tuple[float, float, float]]]:
    """
    Introspect hazard geometry (center + radius) from the environment.

    Safety-Gymnasium exposes hazard geometry through different attribute
    paths depending on version. We try each path in order of preference, and
    return None (never raise) on failure, since a shield with no hazards is a
    legitimate, if inert, configuration -- unlike a failed position lookup,
    which corrupts every subsequent check.

    Must be called every episode, not just once at shield construction:
    Safety-Gymnasium re-randomizes the hazard layout on every reset, and a
    shield holding the episode-0 layout checks the wrong geometry for
    ~999 of every 1000 steps at typical episode lengths (this was D2).

    :param env: The (possibly wrapped) environment; env.unwrapped is used.
        :type env: gym.Env

    :return: List of (x, y, radius) hazard discs, or None if none were found.
        :rtype: Optional[List[Tuple[float, float, float]]]
    """
    try:
        uw = env.unwrapped

        hazards_pos = None
        hazards_size = 0.2  # default radius for point tasks in Safety-Gymnasium

        if hasattr(uw, "task") and hasattr(uw.task, "hazards"):
            h = uw.task.hazards
            hazards_pos = getattr(h, "pos", None)
            hazards_size = float(getattr(h, "size", 0.2))
        elif hasattr(uw, "task") and hasattr(uw.task, "hazards_pos"):
            # Older Safety-Gymnasium layout where pos lives directly on task
            hazards_pos = uw.task.hazards_pos
            hazards_size = float(getattr(uw.task, "hazards_size", 0.2))
        elif hasattr(uw, "world") and hasattr(uw.world, "hazards_pos"):
            # Legacy Safety-Gym (pre-0.4) attribute path
            hazards_pos = uw.world.hazards_pos
            hazards_size = float(getattr(uw.world, "hazards_size", 0.2))

        if hazards_pos is not None and len(hazards_pos) > 0:
            # Hazard positions are 3D (x, y, z); the shield only needs x, y
            return [(float(p[0]), float(p[1]), hazards_size) for p in hazards_pos]

    except Exception as e:
        print(f"[Shield] WARNING: hazard introspection failed: {e}")

    return None


def refresh_shield_hazards(env: gym.Env, shield: Any) -> None:
    """
    Re-resolve hazards from env and push them into shield.

    Thin wrapper around resolve_hazards that logs the outcome the same way
    at every call site (construction and every reset), so the two are never
    allowed to drift into printing different messages for the same failure.

    :param env: The (possibly wrapped) environment.
        :type env: gym.Env
    :param shield: Shield object exposing set_hazards(list).
        :type shield: Any

    :return: None.
        :rtype: None
    """
    hz = resolve_hazards(env)
    if hz:
        shield.set_hazards(hz)
        print(f"[Shield] loaded {len(hz)} hazards: {hz}")
    else:
        print("[Shield] WARNING: could not find hazard positions, shield is pass-through")


class ShieldingActionWrapper(gym.ActionWrapper):
    """
    Project actions through a safety shield before passing them to the env.

    The shield is any object that implements:

        safe_action = shield.step(action, {"agent_pos": xy})

    The wrapper resolves the agent's real position from env.unwrapped at
    every step (see resolve_agent_pos) rather than from the raw observation,
    and refreshes hazard geometry from env.unwrapped on every reset (see
    refresh_shield_hazards), since Safety-Gymnasium re-randomizes hazards
    per episode. It replaces the original action with safe_action, then logs
    whether the shield intervened and by how much via
    info["shield_intervened"] / info["shield_deflection_magnitude"]. It also
    normalizes the step output so info["cost"] is always present, mirroring
    the behavior of CostInfoWrapper and RewardShapingWrapper.
    """

    def __init__(self, env: gym.Env, shield: Any):
        """
        Initialize the shielding action wrapper.

        :param env: Base environment whose actions should be filtered.
            :type env: gym.Env
        :param shield: Safety shield object with a step(action, obs) interface.
            May optionally expose a boolean attribute last_intervened and an
            on_reset() method.
            :type shield: Any
        """
        super().__init__(env)
        self.shield = shield
        self._last_obs = None  # type: Any

    def step_with_cost(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step the wrapped environment and ensure info["cost"] exists.

        Normalizes both the 6-tuple Safety-Gymnasium layout and the 5-tuple
        Gymnasium layout to a standard 5-tuple.

        :param action: Action to pass to the wrapped environment.
            :type action: Any

        :return: Standard 5-tuple (obs, reward, terminated, truncated, info)
            with info["cost"] always present.
            :rtype: Tuple[Any, float, bool, bool, Dict[str, Any]]
        """
        out = self.env.step(action)
        if not isinstance(out, tuple):
            raise RuntimeError(f"env.step returned non-tuple: {out!r}")

        if len(out) == 6:
            # Safety-Gymnasium style: (obs, reward, cost, terminated, truncated, info)
            obs, reward, cost, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            info["cost"] = float(cost)
        elif len(out) == 5:
            # Gymnasium style: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = out
            info = dict(info) if info is not None else {}
            info.setdefault("cost", float(info.get("cost", 0.0)))
        else:
            raise RuntimeError(f"Unexpected env.step tuple length: {len(out)}")

        return obs, float(reward), bool(terminated), bool(truncated), info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Apply the safety shield to the proposed action, then step the env.

        The agent's real position is resolved from env.unwrapped at step
        time (not read from the raw observation -- see resolve_agent_pos and
        D1) and packaged the way the shield expects. After stepping, last_obs
        is updated and the shield's intervention flag and deflection
        magnitude are surfaced in info.

        :param action: Proposed action from the policy.
            :type action: Any

        :return: Standard 5-tuple (obs, reward, terminated, truncated, info)
            where info["shield_intervened"] indicates whether the shield
            modified the original action.
            :rtype: Tuple[Any, float, bool, bool, Dict[str, Any]]
        """
        agent_pos = resolve_agent_pos(self.env)
        safe_action = self.shield.step(action, {"agent_pos": agent_pos})
        obs, reward, terminated, truncated, info = self.step_with_cost(safe_action)
        self._last_obs = obs

        # Surface the shield's intervention flag so the logger can track it
        last_intervened = getattr(self.shield, "last_intervened", None)
        if last_intervened is not None:
            info["shield_intervened"] = bool(last_intervened)
        else:
            info.setdefault("shield_intervened", False)

        # Position before this step and the XY action actually executed
        # (post-shield, whether or not it intervened). Together with the
        # resolved position on the NEXT step, this is what lets an external
        # caller test the shield's own kinematic assumption
        # (next_pos = pos + dt*a_xy) against the real environment dynamics --
        # e.g. to check whether that assumption holds for a nonholonomic
        # robot the way it does for a holonomic one. Not used internally;
        # exposed purely as instrumentation.
        info["shield_agent_pos"] = np.asarray(agent_pos, dtype=np.float32).copy()
        info["shield_safe_action_xy"] = np.asarray(
            safe_action[:2], dtype=np.float32
        ).copy()

        # Total action change from the original proposed action, across every
        # mechanism the shield applied this step (gradient deflection AND any
        # bisection fallback, for RiemannianShield; the single bisection
        # projection for GenericKeepoutShield).
        info["shield_deflection_magnitude"] = float(
            getattr(self.shield, "last_deflection_magnitude", 0.0)
        )

        # Gradient-stage-only magnitude and whether its norm clip fired this
        # step, present only for shields with a gradient stage (RiemannianShield).
        # Kept separate from shield_deflection_magnitude above: conflating the
        # two was a bug found in review -- the gradient stage alone is bounded
        # by alpha*max_action_norm, but the total can be larger once the
        # bisection fallback also fires, and averaging them together made
        # "deflection magnitude" not actually measure the gradient mechanism
        # the Riemannian shield is supposed to be characterized by.
        info["shield_gradient_deflection_magnitude"] = float(
            getattr(self.shield, "last_gradient_deflection_magnitude", 0.0)
        )
        info["shield_gradient_clip_fired"] = bool(
            getattr(self.shield, "last_gradient_clip_fired", False)
        )
        info["shield_gradient_intervened"] = bool(
            getattr(self.shield, "last_gradient_intervened", False)
        )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and the shield's internal state.

        Stores the initial observation so the first call to step() has a
        valid obs to pass to the shield. Also re-resolves hazard geometry
        from the freshly-reset environment: Safety-Gymnasium re-randomizes
        hazards on every reset, so without this the shield would keep
        checking the episode-0 layout for the rest of training (D2).

        :param kwargs: Keyword arguments forwarded to the wrapped env's reset.
            :type kwargs: dict

        :return: (obs, info) as returned by the wrapped environment.
            :rtype: Tuple[Any, Dict[str, Any]]
        """
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        # Reset shield episode state (clears intervention counters, etc.)
        if hasattr(self.shield, "on_reset"):
            self.shield.on_reset()
        if hasattr(self.shield, "set_hazards"):
            refresh_shield_hazards(self.env, self.shield)
        return obs, info


def _try_make(env_id: str, seed: int) -> gym.Env:
    """
    Create a raw environment and normalize its step output to a 5-tuple.

    If the env is wrapped in a TimeLimit, normalization is applied inside
    the TimeLimit so that episode truncation semantics are preserved.

    :param env_id: Gymnasium or Safety-Gymnasium environment ID. Prefix with
        "mujoco:" to load a custom MuJoCo model via MujocoRoboticEnv.
        :type env_id: str
    :param seed: Random seed passed to env.reset().
        :type seed: int

    :return: Seeded environment with normalized step output.
        :rtype: gym.Env
    """
    if env_id.startswith("mujoco:"):
        model_path = env_id[len("mujoco:"):]
        env = MujocoRoboticEnv(model_path=model_path)
        env.reset(seed=seed)
        return env

    env = gym.make(env_id, disable_env_checker=True)

    if isinstance(env, TimeLimit):
        # Unwrap the TimeLimit, normalize the inner env, then rewrap so
        # truncation due to step limit is still signalled correctly
        tl = env
        base = NormalizeStepReturn(tl.env)
        max_steps = getattr(tl, "max_episode_steps", getattr(tl, "_max_episode_steps", None))
        env = TimeLimit(base, max_episode_steps=max_steps) if max_steps is not None else TimeLimit(base)

    env.reset(seed=seed)
    return env


def make_env(
    env_id: str,
    seed: int = 0,
    use_shield: bool = False,
    shield_factory: Optional[Callable[[gym.Env], Any]] = None,
    reward_shaping_get_lambda: Optional[Callable[[], float]] = None,
) -> gym.Env:
    """
    Factory for Safe RL environments.

    Wraps a base Gymnasium or Safety-Gymnasium env in the following order:

    1. CostInfoWrapper: guarantees a per-step cost and episodic cost summaries.
    2. RewardShapingWrapper (optional): applies r' = r - lambda * cost.
    3. ShieldingActionWrapper (optional): projects actions through a geometric shield.
    4. Monitor: tracks episode returns and lengths for SB3 logging.

    The wrapper order matters: the shield sees the already-shaped reward, and
    Monitor always sits outermost so SB3 callbacks read consistent episode stats.

    :param env_id: Gymnasium or Safety-Gymnasium environment ID.
        :type env_id: str
    :param seed: Random seed for the environment.
        :type seed: int
    :param use_shield: If True, wraps the env with ShieldingActionWrapper.
        :type use_shield: bool
    :param shield_factory: Factory that receives the cost-wrapped env and returns
        a shield object. Required when use_shield=True.
        :type shield_factory: Optional[Callable[[gym.Env], Any]]
    :param reward_shaping_get_lambda: Callable returning the current lambda for
        Lagrangian reward shaping. If None, no shaping is applied.
        :type reward_shaping_get_lambda: Optional[Callable[[], float]]

    :return: Fully wrapped environment ready for safe RL training.
        :rtype: gym.Env
    """
    env = _try_make(env_id, seed=seed)

    # Always ensure cost is present and episodic summaries are tracked
    env = CostInfoWrapper(env)

    if reward_shaping_get_lambda is not None:
        env = RewardShapingWrapper(env, reward_shaping_get_lambda)

    if use_shield and shield_factory is not None:
        shield = shield_factory(env)
        env = ShieldingActionWrapper(env, shield)

    return Monitor(env)


def is_safety_env(env_id: str) -> bool:
    """
    Heuristic check for whether an env ID belongs to Safety-Gymnasium.

    :param env_id: Gymnasium or Safety-Gymnasium environment ID.
        :type env_id: str

    :return: True if env_id starts with "Safety".
        :rtype: bool
    """
    return env_id.startswith("Safety")
