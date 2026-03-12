from typing import Callable, Optional, Any, Dict, Tuple
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


class ShieldingActionWrapper(gym.ActionWrapper):
    """
    Project actions through a safety shield before passing them to the env.

    The shield is any object that implements:

        safe_action = shield.step(action, obs)

    The wrapper replaces the original action with safe_action, then logs
    whether the shield intervened via info["shield_intervened"]. It also
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

        The most recent observation is passed to the shield so it can make a
        geometry-aware decision. After stepping, last_obs is updated and the
        shield's intervention flag is surfaced in info.

        :param action: Proposed action from the policy.
            :type action: Any

        :return: Standard 5-tuple (obs, reward, terminated, truncated, info)
            where info["shield_intervened"] indicates whether the shield
            modified the original action.
            :rtype: Tuple[Any, float, bool, bool, Dict[str, Any]]
        """
        safe_action = self.shield.step(action, self._last_obs)
        obs, reward, terminated, truncated, info = self.step_with_cost(safe_action)
        self._last_obs = obs

        # Surface the shield's intervention flag so the logger can track it
        last_intervened = getattr(self.shield, "last_intervened", None)
        if last_intervened is not None:
            info["shield_intervened"] = bool(last_intervened)
        else:
            info.setdefault("shield_intervened", False)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and the shield's internal state.

        Stores the initial observation so the first call to step() has a
        valid obs to pass to the shield.

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
