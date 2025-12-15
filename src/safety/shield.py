from typing import List, Tuple, Optional, Any
import numpy as np

class GenericKeepoutShield:
    """
    Conservative geometric keepout shield for 2D robots (e.g. SafetyPoint, SafetyCar).

    The shield maintains a list of circular hazard regions and modifies the agent's
    proposed action when the predicted next position would enter any hazard disc.

    High-level behavior:
    - Extract the agent's current XY position from the observation (when available).
    - Interpret the first two components of the action as an XY velocity command.
    - Predict the next position: pos_next = pos + dt * a_xy.
    - If pos_next lies inside a hazard disc, scale the action magnitude down along
      its direction so that the next position stays just outside the hazard.
    - Otherwise, return the original action unchanged.

    Notes:
    - This is a purely geometric shield, not a full dynamics model. It improves
      safety locally but does not provide formal guarantees in all scenarios.
    - If the agent's position cannot be inferred from the observation, the shield
      acts as a pass-through (no intervention).
    - For 3D tasks such as Fetch manipulation, this 2D shield is not directly
      applicable and will again degenerate to a pass-through.

    Usage example::
        shield = GenericKeepoutShield(hazards=[(0.0, 0.0, 0.5)], dt=0.1, max_action_norm=1.0)
        safe_action = shield.step(action, obs)
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
            the shield starts with no hazards and will act as a pass-through.
        :type hazards: Optional[List[Tuple[float, float, float]]]
        :param dt: Effective time step used to predict the next position
            (pos_next = pos + dt * a_xy).
        :type dt: float
        :param max_action_norm: Maximum allowed L2 norm of the XY action component.
            Actions exceeding this norm are rescaled before safety checks.
        :type max_action_norm: float
        :param epsilon: Small margin used when computing the safe boundary around
            hazards; the shield keeps the predicted position at least `epsilon`
            outside the hazard radius.
        :type epsilon: float

        :return: None
        :rtype: None
        """
        self.hazards: List[Tuple[float, float, float]] = hazards if hazards is not None else []
        self.dt: float = dt
        self.max_action_norm: float = max_action_norm
        self.epsilon: float = epsilon

        # Diagnostics
        self.last_intervened: bool = False
        self.interventions_in_episode: int = 0

    def set_hazards(self, hazards: List[Tuple[float, float, float]]) -> None:
        """
        Update the list of hazard discs.

        :param hazards: New list of hazard discs, each as (x, y, radius).
        :type hazards: List[Tuple[float, float, float]]

        :return: None
        :rtype: None
        """
        self.hazards = hazards

    def on_reset(self) -> None:
        """
        Reset episode-level intervention counters.

        This should be called at the beginning of each new episode.
        It clears both the last_intervened flag and the per-episode
        intervention count.

        :return: None
        :rtype: None
        """
        self.interventions_in_episode = 0
        self.last_intervened = False

    def _extract_agent_xy(self, obs: Any) -> Optional[np.ndarray]:
        """
        Extract the agent's 2D position from an observation.

        The method supports common observation structures used in
        Safety-Gymnasium and robotics tasks:

        - dict observations with key "agent_pos"
        - dict observations with key "achieved_goal"

        If no suitable key is found, the function returns None and the
        shield will act as a pass-through.

        :param obs: Environment observation at the current time step.
        :type obs: Any

        :return: 2D position of the agent as a NumPy array [x, y], or None if
            the position cannot be reliably extracted.
        :rtype: Optional[np.ndarray]
        """
        if obs is None:
            return None

        if isinstance(obs, dict):
            if "agent_pos" in obs:
                pos = obs["agent_pos"]
                return np.array(pos[:2], dtype=np.float32)
            if "achieved_goal" in obs:
                ag = obs["achieved_goal"]
                return np.array(ag[:2], dtype=np.float32)

        # For other observation types, we do not attempt to infer the position.
        return None

    def step(self, action: np.ndarray, obs: Any) -> np.ndarray:
        """
        Project a proposed action through the geometric keepout shield.

        If the predicted next position (based on the current observation and
        proposed action) would violate a hazard constraint, the action is
        rescaled along its direction so that the next position remains just
        outside the hazard. Otherwise, the original action is returned.

        :param action: Proposed action from the policy. The first two entries
            are interpreted as XY velocity components.
        :type action: numpy.ndarray
        :param obs: Current environment observation, used to extract the agent's
            XY position. If the position cannot be extracted, the action is
            returned unchanged.
        :type obs: Any

        :return: Safe action after optional projection. If an intervention occurs,
            the XY components are scaled; otherwise, the original action is
            returned.
        :rtype: numpy.ndarray
        """
        self.last_intervened = False

        # No hazards configured --> nothing to do
        if not self.hazards:
            return action

        pos = self._extract_agent_xy(obs)
        if pos is None:
            # Position not available --> pass-through for safety
            return action

        a = np.array(action, dtype=np.float32)
        if a.shape[0] < 2:
            # Action is not 2D --> leave it unchanged
            return action

        a_xy = a[:2]

        # Clip the action norm to max_action_norm
        norm = np.linalg.norm(a_xy)
        if norm > self.max_action_norm:
            a_xy = a_xy / (norm + 1e-8) * self.max_action_norm

        # Predict next position under the proposed action
        next_pos = pos + self.dt * a_xy

        # Check whether the next position enters any hazard disc
        needs_projection = False
        scale = 1.0
        for (hx, hy, hr) in self.hazards:
            d = np.linalg.norm(next_pos - np.array([hx, hy], dtype=np.float32))
            if d <= hr:
                needs_projection = True
                # Find the largest safe scale t in [0, 1] such that:
                #   ||pos + dt * (t * a_xy) - (hx, hy)|| >= hr - epsilon
                # We approximate this with a simple bisection search on t.
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
