from typing import Callable, Optional, Any, Dict, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit

# Try to import Safety-Gymnasium so that env IDs like "SafetyPointPush1-v0" work
# If it's not installed, gymnasium will raise an error when such envs are created
try:
    import safety_gymnasium  # noqa: F401
except Exception:
    print("WARNING: safety_gymnasium not imported")
    pass


class NormalizeStepReturn(gym.Wrapper):
    """
    Normalize the output of `env.step` to the Gymnasium 5-tuple:

        (obs, reward, terminated, truncated, info)

    This wrapper handles several common layouts:

    - 4-tuple: (obs, reward, done, info)
    - 5-tuple: (obs, reward, terminated, truncated, info)
    - 6-tuple (safety format): (obs, reward, cost, terminated, truncated, info)
    - 6-tuple (extra payload): (obs, reward, terminated, truncated, info, extra)

    In the 6-tuple safety case, the scalar `cost` is moved into `info["cost"]`.
    In the extra-payload case, the extra fields are merged into `info`.
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
                # Merge extra keys, renaming in case of collision
                for k, v in info_extra.items():
                    key = f"extra_{k}" if k in info else k
                    info[key] = v
            else:
                info["extra"] = info_extra
            return obs, reward, bool(terminated), bool(truncated), info

        raise RuntimeError(f"Unexpected env.step tuple length: {n}")

class CostInfoWrapper(gym.Wrapper):
    """
    Ensure that each step exposes a scalar `cost` in the info dict and track
    episodic cost/length.

    - If the wrapped env already provides cost (e.g. Safety-Gymnasium), it is
      reused.
    - Otherwise, cost defaults to 0.0.

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
        Single-step helper that normalizes the env output and ensures `info["cost"]` exists.

        Accepts:
        - (obs, reward, cost, terminated, truncated, info)
        - (obs, reward, terminated, truncated, info)

        Returns a 5-tuple:
            (obs, reward, terminated, truncated, info)
        where info["cost"] is always a float.
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

        # Attach episodic summaries when the episode ends
        if terminated or truncated:
            info["episodic_cost"] = self.episodic_cost
            info["episodic_length"] = self.steps
        return obs, rew, terminated, truncated, info


class RewardShapingWrapper(gym.RewardWrapper):
    """
    Lagrangian reward shaping for constrained RL:

        r_shaped = reward - lambda * cost

    The current lambda value is provided by the callable `get_lambda`, which
    allows online adaptation (e.g. dual gradient updates in LagPPO/RCPO).

    The wrapper:
    - ensures a "cost" field is present in info (if the env supports it),
    - replaces the reward with r_shaped,
    - logs "shaped_reward" and "lambda" in the info dict.
    """

    def __init__(self, env: gym.Env, get_lambda: Callable[[], float]):
        super().__init__(env)
        self.get_lambda = get_lambda

    def step_with_cost(self, action):
        """
        Normalize the env output and ensure info["cost"] exists.
        Mirrors the logic in CostInfoWrapper, but kept local to avoid
        changing the wrapped env order here.
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
    Action wrapper that projects actions through a safety shield before
    passing them to the environment.

    The shield is a callable object with a method:

        safe_action = shield.step(action, obs)

    where `obs` is the most recent observation (or ``None`` if not available).
    The wrapper replaces the original action with `safe_action` in the call to
    ``env.step`` and logs whether the shield intervened in the info dict via
    ``info["shield_intervened"]``.

    In addition, this wrapper normalizes the step output so that a scalar
    safety cost is always exposed as ``info["cost"]`` (mirroring the behavior
    of CostInfoWrapper/RewardShapingWrapper).
    """

    def __init__(self, env: gym.Env, shield: Any):
        """
        Initialize the shielding action wrapper.

        :param env: Base Gymnasium or Safety-Gymnasium environment whose actions
            should be filtered by the safety shield.
        :type env: gym.Env
        :param shield: Safety shield object that exposes a ``step(action, obs)``
            interface. The shield must accept the proposed action and the latest
            observation, and return a (possibly modified) safe action. It may
            optionally expose a boolean attribute ``last_intervened`` indicating
            whether it modified the action at the last call, and a ``on_reset``
            method that will be called when the environment is reset.
        :type shield: Any
        """
        super().__init__(env)
        self.shield = shield
        self._last_obs = None  # type: Any

    def step_with_cost(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Take a single step in the wrapped environment, ensuring that the
        returned info dict contains a scalar safety cost under the key "cost".

        The wrapped environment is expected to return either:

        - a 6-tuple ``(obs, reward, cost, terminated, truncated, info)``, or
        - a 5-tuple ``(obs, reward, terminated, truncated, info)`` where cost
          may or may not already be present in ``info``.

        This helper normalizes both cases to the standard 5-tuple and guarantees
        that ``info["cost"]`` is present (defaulting to 0.0 if missing).

        :param action: Action to be passed to the wrapped environment (after
            optional shielding). Its type should match the environment's action
            space, typically a numpy array for continuous control.
        :type action: Any
        :return: A 5-tuple ``(obs, reward, terminated, truncated, info)`` where
            ``info["cost"]`` is a float representing the instantaneous safety
            cost at this step.
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
        Apply the safety shield to the proposed action and step the environment.

        This method:

        1. Uses the most recent observation (if available) together with the
           proposed action to compute a shielded action via
           ``safe_action = shield.step(action, obs)``.
        2. Passes the shielded action to the wrapped environment and normalizes
           the output via `step_with_cost`.
        3. Updates ``self._last_obs`` to the new observation.
        4. Logs whether the shield intervened at this step via
           ``info["shield_intervened"]``, assuming the shield exposes a
           boolean attribute ``last_intervened``.

        :param action: Proposed action from the policy or agent. The shield may
            modify this action if it predicts a safety violation.
        :type action: Any
        :return: A 5-tuple ``(obs, reward, terminated, truncated, info)`` where
            the transition corresponds to the **shielded** action, ``info["cost"]``
            contains the safety cost, and ``info["shield_intervened"]`` indicates
            whether the shield modified the original action.
        :rtype: Tuple[Any, float, bool, bool, Dict[str, Any]]
        """
        # Use the last observation (if available) to compute a shielded action
        if self._last_obs is None:
            safe_action = self.shield.step(action, None)
        else:
            safe_action = self.shield.step(action, self._last_obs)

        obs, reward, terminated, truncated, info = self.step_with_cost(safe_action)
        self._last_obs = obs

        # If the shield tracks whether it intervened, surface that in info.
        last_intervened = getattr(self.shield, "last_intervened", None)
        if last_intervened is not None:
            info["shield_intervened"] = bool(last_intervened)
        else:
            info.setdefault("shield_intervened", False)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and the internal shield state.

        This method:

        - Calls ``env.reset(**kwargs)`` on the wrapped environment.
        - Stores the initial observation as ``self._last_obs`` so that the
          next call to `step` can pass a valid observation to the shield.
        - If the shield implements an ``on_reset`` method, it is called to
          reset any internal shield state (e.g. memory of previous hazards).

        :param kwargs: Additional keyword arguments forwarded to the wrapped
            environment's ``reset`` method (e.g. seed, options).
        :type kwargs: dict
        :return: A tuple ``(obs, info)`` as returned by the wrapped environment.
        :rtype: Tuple[Any, Dict[str, Any]]
        """
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        # If the shield exposes a reset hook, call it
        if hasattr(self.shield, "on_reset"):
            self.shield.on_reset()
        return obs, info

def _try_make(env_id: str, seed: int) -> gym.Env:
    """
    Create a raw environment instance from an env_id and seed, then ensure
    its `step` method returns a normalized 5-tuple via NormalizeStepReturn.

    If the underlying environment is wrapped in a TimeLimit, normalization
    is applied *inside* the TimeLimit so that episode truncation semantics
    are preserved.
    """
    env = gym.make(env_id, disable_env_checker=True)

    if isinstance(env, TimeLimit):
        # Unwrap the TimeLimit, normalize the inner env, then rewrap.
        tl = env
        base = tl.env  # unwrap
        base = NormalizeStepReturn(base)
		# Recreate TimeLimit preserving its max steps
        max_steps = getattr(tl, "max_episode_steps", getattr(tl, "_max_episode_steps", None))
        env = TimeLimit(base, max_episode_steps=max_steps) if max_steps is not None else TimeLimit(base)
    # If there is no TimeLimit, we could still wrap with NormalizeStepReturn,
    # but many modern envs already follow the 5-tuple API. We keep the logic
    # minimal here and rely on downstream wrappers to handle cost layout.

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

    It creates a base Gymnasium / Safety-Gymnasium env and wraps it with:

    1. CostInfoWrapper
       - guarantees a per-step scalar cost and episodic cost summaries.

    2. RewardShapingWrapper (optional)
       - applies Lagrangian-style shaping: r' = r - lambda * cost,
         where lambda is provided by `reward_shaping_get_lambda`.

    3. ShieldingActionWrapper (optional)
       - projects actions through a safety shield constructed by `shield_factory`.

    Parameters:
    
    :param env_id: Environment ID (e.g., "SafetyPointPush1-v0")
		:type env_id: str
    :param seed: Random seed for the environment
		:type seed: int, default=0
    :param use_shield: If True, wraps the env with a shielding action wrapper
		:type use_shield: bool, default=False
    :param shield_factory: Factory that receives the (already cost-wrapped) env and returns a shield object
		:type shield_factory: Callable[[gym.Env], Any], optional
    :param reward_shaping_get_lambda: Callable returning the current lambda value for reward shaping
		:type reward_shaping_get_lambda: Callable[[], float], optional
       
    :return: A wrapped environment suitable for safe RL algorithms
		:rtype: gym.Env
    """
    env = _try_make(env_id, seed=seed)

    # Always ensure cost is present and episodic summaries are tracked.
    env = CostInfoWrapper(env)

    # Optional Lagrangian reward shaping
    if reward_shaping_get_lambda is not None:
        env = RewardShapingWrapper(env, reward_shaping_get_lambda)

    # Optional action shielding
    if use_shield and shield_factory is not None:
        shield = shield_factory(env)
        env = ShieldingActionWrapper(env, shield)

    return env


def is_safety_env(env_id: str) -> bool:
    """
    Heuristic helper to check whether an environment is a Safety-Gymnasium env.
    """
    return env_id.startswith("Safety")