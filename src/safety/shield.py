from typing import List, Optional, Tuple, Any
import numpy as np


class GenericKeepoutShield:
    """
    Conservative geometric keepout shield for 2D robots (e.g. SafetyPoint, SafetyCar).

    The shield maintains a list of circular hazard regions and modifies the
    agent's proposed action when the predicted next position would enter any
    hazard disc. It operates entirely at execution time and requires no
    training: the only inputs are hazard positions and radii, which are read
    from the environment at the start of each episode.

    High-level behavior:
    - Extract the agent's current XY position from the observation.
    - Interpret the first two components of the action as an XY velocity command.
    - Predict the next position: pos_next = pos + dt * a_xy.
    - If pos_next lies inside a hazard disc, scale the action magnitude down
      along its current direction until the next position stays just outside
      the hazard boundary.
    - Otherwise, return the original action unchanged.

    This is a purely geometric shield. It improves safety locally but does not
    provide formal guarantees in all scenarios. If the agent's position cannot
    be inferred from the observation, the shield acts as a pass-through.
    For 3D tasks such as Fetch manipulation, it also degenerates to a
    pass-through since the 2D projection is not meaningful.
    """

    def __init__(
        self,
        hazards: Optional[List[Tuple[float, float, float]]] = None,
        dt: float = 0.1,
        max_action_norm: float = 1.0,
        epsilon: float = 1e-3,
    ) -> None:
        """
        Initialize the keepout shield.

        :param hazards: List of hazard discs, each as (x, y, radius). If None,
            the shield starts with no hazards and acts as a pass-through.
            :type hazards: Optional[List[Tuple[float, float, float]]]
        :param dt: Time step used to predict the next position
            (pos_next = pos + dt * a_xy).
            :type dt: float
        :param max_action_norm: Maximum allowed L2 norm of the XY action
            component. Actions exceeding this are rescaled before safety checks.
            :type max_action_norm: float
        :param epsilon: Safety margin kept between the predicted position and
            the hazard boundary.
            :type epsilon: float

        :return: None.
            :rtype: None
        """
        self.hazards: List[Tuple[float, float, float]] = hazards if hazards is not None else []
        self.dt: float = dt
        self.max_action_norm: float = max_action_norm
        self.epsilon: float = epsilon

        # Diagnostics: readable by the environment wrapper to log intervention stats
        self.last_intervened: bool = False
        self.interventions_in_episode: int = 0

    def set_hazards(self, hazards: List[Tuple[float, float, float]]) -> None:
        """
        Update the list of hazard discs.

        Called at the start of each episode so the shield always has the
        current hazard layout, which may change between episodes.

        :param hazards: New list of hazard discs, each as (x, y, radius).
            :type hazards: List[Tuple[float, float, float]]

        :return: None.
            :rtype: None
        """
        self.hazards = hazards

    def on_reset(self) -> None:
        """
        Reset episode-level intervention counters.

        Should be called at the start of each new episode to clear
        per-episode statistics before the next rollout.

        :return: None.
            :rtype: None
        """
        self.interventions_in_episode = 0
        self.last_intervened = False

    def _extract_agent_xy(self, obs: Any) -> Optional[np.ndarray]:
        """
        Extract the agent's 2D position from an observation.

        Supports the observation layouts used in Safety-Gymnasium and
        Gymnasium-Robotics: dict observations with "agent_pos" or
        "achieved_goal", and flat numpy arrays where XY are at indices 0:2
        (the layout used by SafetyPointPush1-v0).

        Returns None if the position cannot be reliably extracted, which
        causes the shield to act as a pass-through for that step.

        :param obs: Environment observation at the current time step.
            :type obs: Any

        :return: 2D position [x, y], or None if extraction fails.
            :rtype: Optional[np.ndarray]
        """
        if obs is None:
            return None

        if isinstance(obs, dict):
            if "agent_pos" in obs:
                return np.array(obs["agent_pos"][:2], dtype=np.float32)
            if "achieved_goal" in obs:
                return np.array(obs["achieved_goal"][:2], dtype=np.float32)

        # Flat array: SafetyPointPush1-v0 puts agent XY at indices 0:2
        if isinstance(obs, np.ndarray) and obs.shape[0] >= 2:
            return np.array(obs[:2], dtype=np.float32)

        return None

    def step(self, action: np.ndarray, obs: Any) -> np.ndarray:
        """
        Project a proposed action through the geometric keepout shield.

        If the predicted next position would violate a hazard constraint,
        the action is rescaled along its current direction using bisection
        so the next position stays just outside the hazard boundary.
        If multiple hazards are violated, the most conservative scale is used.

        :param action: Proposed action from the policy. The first two entries
            are interpreted as XY velocity components.
            :type action: np.ndarray
        :param obs: Current environment observation, used to extract the
            agent's XY position. If the position cannot be extracted, the
            action is returned unchanged.
            :type obs: Any

        :return: Safe action after optional scaling. Returns the original
            action if no intervention is needed or if position is unavailable.
            :rtype: np.ndarray
        """
        self.last_intervened = False

        # No hazards configured: nothing to check
        if not self.hazards:
            return action

        pos = self._extract_agent_xy(obs)
        if pos is None:
            return action

        a = np.array(action, dtype=np.float32)
        if a.shape[0] < 2:
            return action

        a_xy = a[:2]

        # Clip to max_action_norm before predicting the next position
        norm = np.linalg.norm(a_xy)
        if norm > self.max_action_norm:
            a_xy = a_xy / (norm + 1e-8) * self.max_action_norm

        next_pos = pos + self.dt * a_xy

        # Find the most conservative safe scale across all violated hazards
        needs_projection = False
        scale = 1.0
        for (hx, hy, hr) in self.hazards:
            d = np.linalg.norm(next_pos - np.array([hx, hy], dtype=np.float32))
            if d <= hr:
                needs_projection = True
                # Bisection: find the largest t in [0, 1] such that
                #   ||pos + dt * (t * a_xy) - hazard_center|| >= hr - epsilon
                # 20 iterations gives precision ~1e-6, which is more than enough
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    test_next = pos + self.dt * (mid * a_xy)
                    if np.linalg.norm(test_next - np.array([hx, hy], dtype=np.float32)) <= (hr - self.epsilon):
                        hi = mid
                    else:
                        lo = mid
                scale = min(scale, lo)

        if needs_projection:
            a_proj = a.copy()
            a_proj[:2] = a_xy * scale
            self.last_intervened = True
            self.interventions_in_episode += 1
            return a_proj

        return action
