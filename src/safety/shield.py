import math
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
    provide formal guarantees in all scenarios. The agent's position must be
    supplied by the caller (see _extract_agent_xy) -- it is not recoverable
    from Safety-Gymnasium's flat sensor+lidar observation, so a caller that
    fails to resolve it gets a loud error rather than a silent pass-through.
    """

    def __init__(
        self,
        hazards: Optional[List[Tuple[float, float, float]]] = None,
        dt: float = 0.1,
        max_action_norm: float = 1.0,
        epsilon: float = 1e-3,
        kinematic_model: str = "world_xy",
        body_frame_M: Optional[np.ndarray] = None,
        body_frame_b: Optional[np.ndarray] = None,
        interior_override: bool = False,
    ) -> None:
        """
        Initialize the keepout shield.

        :param hazards: List of hazard discs, each as (x, y, radius). If None,
            the shield starts with no hazards and acts as a pass-through.
            :type hazards: Optional[List[Tuple[float, float, float]]]
        :param dt: Time step used to predict the next position under the
            default "world_xy" kinematic model
            (pos_next = pos + dt * a_xy). Unused under "heading_fit".
            :type dt: float
        :param max_action_norm: Maximum allowed L2 norm of the XY action
            component. Actions exceeding this are rescaled before safety checks.
            :type max_action_norm: float
        :param epsilon: Safety margin kept between the predicted position and
            the hazard boundary.
            :type epsilon: float
        :param kinematic_model: Which next-position prediction to use.
            "world_xy" (default, unchanged behavior) treats the action as a
            world-frame xy velocity: pos_next = pos + dt * a_xy. Measured
            against ground truth (SESSION_HANDOFF.md, 2026-09-11), this
            assumption's direction cosine similarity with the true
            displacement is statistically indistinguishable from zero on
            BOTH SafetyPointGoal1-v0 and SafetyCarGoal1-v0 -- a universal
            defect, not a robot-specific one. "heading_fit" instead predicts
            pos_next = pos + R(heading) @ (body_frame_M @ a_xy +
            body_frame_b), a heading-relative (turn-and-drive) model fit
            from rollout data (see scripts/fit_kinematic_model.py), which on
            held-out data cuts median prediction error from 43-57% of the
            hazard radius to 4-6% on both robots. Requires heading to be
            supplied via obs["heading"] at step time (raises if absent) and
            requires body_frame_M/body_frame_b to be set.
            :type kinematic_model: str
        :param body_frame_M: 2x2 matrix mapping action to body-frame
            displacement under "heading_fit" (disp_body = M @ a_xy + b).
            Required when kinematic_model == "heading_fit".
            :type body_frame_M: Optional[np.ndarray]
        :param body_frame_b: 2-vector intercept for the body-frame
            displacement fit. Required when kinematic_model == "heading_fit"
            or interior_override is True.
            :type body_frame_b: Optional[np.ndarray]
        :param interior_override: If True, whenever the agent's CURRENT
            position (not a predicted one) is already inside a hazard
            (clearance <= 0), the proposed action is discarded entirely --
            not blended -- and replaced with a full-magnitude
            (max_action_norm) action chosen to maximize outward body-frame
            displacement, per body_frame_M, away from the
            deepest-penetrating hazard. This tests whether the interior
            trap documented in SESSION_HANDOFF.md (car dwell 654-810 steps
            vs point 15-19) is a consequence of the shield BLENDING a weak
            correction with the policy's own (often inward-pointing)
            action, or is inherent to filtering the action space at all --
            see _interior_override_action. Independent of kinematic_model:
            the normal (non-interior) prediction path is unaffected: this
            only replaces what happens once already violating.
            :type interior_override: bool

        :raises ValueError: If kinematic_model is "heading_fit" but
            body_frame_M or body_frame_b is not supplied, if
            kinematic_model is neither "world_xy" nor "heading_fit", or if
            interior_override is True without body_frame_M/body_frame_b.
        """
        self.hazards: List[Tuple[float, float, float]] = hazards if hazards is not None else []
        self.dt: float = dt
        self.max_action_norm: float = max_action_norm
        self.epsilon: float = epsilon

        if kinematic_model not in ("world_xy", "heading_fit"):
            raise ValueError(
                f"Unknown kinematic_model {kinematic_model!r}; expected "
                "'world_xy' or 'heading_fit'."
            )
        needs_body_frame_fit = kinematic_model == "heading_fit" or interior_override
        if needs_body_frame_fit and (body_frame_M is None or body_frame_b is None):
            raise ValueError(
                "kinematic_model='heading_fit' and/or interior_override=True "
                "require body_frame_M and body_frame_b (fit via "
                "scripts/fit_kinematic_model.py) -- failing loudly rather "
                "than silently falling back to the known-wrong world_xy "
                "assumption or an untargeted override direction."
            )
        self.kinematic_model: str = kinematic_model
        self.body_frame_M: Optional[np.ndarray] = (
            np.asarray(body_frame_M, dtype=np.float32) if body_frame_M is not None else None
        )
        self.body_frame_b: Optional[np.ndarray] = (
            np.asarray(body_frame_b, dtype=np.float32) if body_frame_b is not None else None
        )
        self.interior_override: bool = interior_override

        # Diagnostics: readable by the environment wrapper to log intervention stats
        self.last_intervened: bool = False
        self.interventions_in_episode: int = 0
        self.last_deflection_magnitude: float = 0.0
        self.last_interior_override_fired: bool = False

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
        self.last_deflection_magnitude = 0.0
        self.last_interior_override_fired = False

    def _extract_agent_xy(self, obs: Any) -> np.ndarray:
        """
        Extract the agent's 2D position from an observation.

        Supports dict observations with "agent_pos" or "achieved_goal" keys.
        There is deliberately no flat-array fallback: Safety-Gymnasium's flat
        observation vector is proprioceptive sensors plus pseudo-lidar and
        contains no absolute position at all, so obs[:2] there is the first
        two accelerometer components, not a position (this was D1 -- the
        shield silently read accelerometer noise as XY for every reported
        run). The real position has to come from the environment
        (env.unwrapped), which is the caller's job: ShieldingActionWrapper
        resolves it at step time and packages it as {"agent_pos": ...}
        before calling into the shield.

        Raises if the position cannot be extracted, rather than returning
        None, so a broken wiring fails loudly instead of silently degrading
        every shielded run to a pass-through (the second half of D1).

        :param obs: Environment observation at the current time step, expected
            to be a dict carrying "agent_pos" or "achieved_goal".
            :type obs: Any

        :return: 2D position [x, y].
            :rtype: np.ndarray

        :raises ValueError: If obs is not a dict carrying a usable position key.
        """
        if isinstance(obs, dict):
            if "agent_pos" in obs:
                return np.array(obs["agent_pos"][:2], dtype=np.float32)
            if "achieved_goal" in obs:
                return np.array(obs["achieved_goal"][:2], dtype=np.float32)

        raise ValueError(
            "Could not extract agent position: expected a dict obs with an "
            f"'agent_pos' or 'achieved_goal' key, got {type(obs).__name__}. "
            "The shield needs a real position from env.unwrapped, supplied by "
            "the caller -- it cannot be recovered from Safety-Gymnasium's flat "
            "sensor+lidar observation."
        )

    def _extract_heading(self, obs: Any) -> Optional[float]:
        """
        Extract the agent's current heading (yaw, radians) from obs, if present.

        Only required when kinematic_model == "heading_fit". Returns None
        rather than raising when absent, so callers using the default
        "world_xy" model (which never needs heading) are unaffected; step()
        itself raises if heading_fit needs it and it is missing.

        :param obs: Environment observation, expected to be a dict optionally
            carrying a "heading" key (yaw in radians).
            :type obs: Any

        :return: Yaw in radians, or None if not present.
            :rtype: Optional[float]
        """
        if isinstance(obs, dict) and "heading" in obs and obs["heading"] is not None:
            return float(obs["heading"])
        return None

    def _predict_displacement(self, a_xy: np.ndarray, heading: Optional[float]) -> np.ndarray:
        """
        Predict world-frame displacement for a given XY action, under
        whichever kinematic_model this shield was configured with.

        "world_xy" (default): dt * a_xy, i.e. the action is a world-frame
        velocity command, independent of heading.

        "heading_fit": R(heading) @ (body_frame_M @ a_xy + body_frame_b), a
        heading-relative model fit from rollout data. Only the M @ a_xy term
        is meant to scale with a partially-applied action during the
        bisection search below; body_frame_b is the residual/drift term
        measured at the action actually taken and is added unscaled,
        matching how it was fit (see scripts/fit_kinematic_model.py).

        :param a_xy: XY action component (possibly scaled by a bisection
            trial factor).
            :type a_xy: np.ndarray
        :param heading: Current yaw in radians. Required (non-None) under
            "heading_fit".
            :type heading: Optional[float]

        :return: Predicted world-frame displacement, shape (2,).
            :rtype: np.ndarray

        :raises ValueError: If kinematic_model == "heading_fit" and heading
            is None.
        """
        if self.kinematic_model == "world_xy":
            return self.dt * a_xy

        if heading is None:
            raise ValueError(
                "kinematic_model='heading_fit' requires a heading, but obs "
                "carried none. The caller must resolve it from env.unwrapped "
                "(see resolve_heading in make_env.py) and pass it as "
                "obs['heading'] -- failing loudly rather than silently "
                "falling back to the known-wrong world_xy assumption."
            )
        disp_body = self.body_frame_M @ a_xy + self.body_frame_b
        c, s = math.cos(heading), math.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return R @ disp_body

    def _interior_override_action(
        self, pos: np.ndarray, heading: Optional[float], a: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Full-authority replacement action for when the agent is ALREADY
        inside a hazard (current clearance <= 0), or None if it is not.

        Ignores the proposed action `a` entirely (this is a replacement, not
        a blend). Picks the deepest-penetrating hazard (most negative
        clearance) and computes the world-frame direction directly away from
        its center, u = (pos - center) / ||pos - center||. Then, using the
        fitted heading-relative model (body_frame_M, rotated by heading),
        inverts it to find the FULL-MAGNITUDE action (norm == max_action_norm,
        not alpha*max_action_norm) whose predicted body-frame displacement is
        best aligned with u:

            target_body = R(heading)^T @ u          (desired direction, body frame)
            a* = max_action_norm * (M^T @ target_body) / ||M^T @ target_body||

        a* maximizes (M @ a) . target_body over all a with ||a|| ==
        max_action_norm (a standard linear-functional-over-a-ball argument),
        i.e. it is the full-strength action that best drives the fitted model
        toward the correct escape direction, given the body-frame gain matrix
        M actually observed for this robot.

        :param pos: Current 2D agent position.
            :type pos: np.ndarray
        :param heading: Current yaw in radians. Required (raises if None).
            :type heading: Optional[float]
        :param a: Original proposed action (only its non-xy tail, if any, is
            preserved; components [:2] are fully replaced).
            :type a: np.ndarray

        :return: The override action, or None if the agent is not currently
            inside any hazard.
            :rtype: Optional[np.ndarray]

        :raises ValueError: If heading is None (interior_override requires it
            regardless of kinematic_model, since it always uses the
            heading-relative model to invert for the escape direction).
        """
        violated = []
        for (hx, hy, hr) in self.hazards:
            d = float(np.linalg.norm(pos - np.array([hx, hy], dtype=np.float32)))
            clearance = d - hr
            if clearance <= 0.0:
                violated.append((clearance, hx, hy, d))
        if not violated:
            return None

        violated.sort(key=lambda t: t[0])  # most negative (deepest) first
        _, hx, hy, d = violated[0]

        if d > 1e-6:
            u = (pos - np.array([hx, hy], dtype=np.float32)) / d
        else:
            u = np.array([1.0, 0.0], dtype=np.float32)  # degenerate: exactly at center

        if heading is None:
            raise ValueError(
                "interior_override requires a heading (see resolve_heading in "
                "make_env.py), regardless of kinematic_model, to invert the "
                "body-frame model for the escape direction -- got None."
            )
        c, s = math.cos(heading), math.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        target_body = R.T @ u

        g = self.body_frame_M.T @ target_body
        g_norm = float(np.linalg.norm(g))
        if g_norm > 1e-8:
            a_xy_override = self.max_action_norm * g / g_norm
        else:
            # Degenerate fit (M^T @ target_body ~= 0): fall back to commanding
            # the desired body-frame direction directly, at full magnitude,
            # rather than dividing by ~0.
            a_xy_override = self.max_action_norm * target_body / (
                np.linalg.norm(target_body) + 1e-8
            )

        override = np.array(a, dtype=np.float32).copy()
        override[:2] = a_xy_override
        return override

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
            agent's XY position. Must carry a real position (see
            _extract_agent_xy); this raises rather than passing the action
            through unchanged if it does not.
            :type obs: Any

        :return: Safe action after optional scaling. Returns the original
            action if no intervention is needed.
            :rtype: np.ndarray
        """
        self.last_intervened = False
        self.last_deflection_magnitude = 0.0
        self.last_interior_override_fired = False

        # No hazards configured: nothing to check. This is the only
        # legitimate pass-through path -- position extraction below always
        # either succeeds or raises.
        if not self.hazards:
            return action

        pos = self._extract_agent_xy(obs)
        heading = self._extract_heading(obs)

        a = np.array(action, dtype=np.float32)
        if a.shape[0] < 2:
            return action

        # FULL-AUTHORITY INTERIOR OVERRIDE: if the agent's CURRENT position
        # (not a predicted one) is already inside a hazard, discard the
        # proposed action entirely -- do not blend it with the override, and
        # do not run the normal predict-then-bisect logic below at all, since
        # that logic's own bisection is what degenerates to scale~=0 in this
        # regime (see SESSION_HANDOFF.md's escape-dynamics finding). This
        # takes priority over everything else in this method, including any
        # gradient-stage deflection a RiemannianShield subclass already
        # applied to `action` before calling here.
        if self.interior_override:
            override = self._interior_override_action(pos, heading, a)
            if override is not None:
                self.last_intervened = True
                self.last_interior_override_fired = True
                self.interventions_in_episode += 1
                self.last_deflection_magnitude = float(
                    np.linalg.norm(override[:2] - a[:2])
                )
                return override

        a_xy = a[:2]

        # Clip to max_action_norm before predicting the next position
        norm = np.linalg.norm(a_xy)
        if norm > self.max_action_norm:
            a_xy = a_xy / (norm + 1e-8) * self.max_action_norm

        next_pos = pos + self._predict_displacement(a_xy, heading)

        # Find the most conservative safe scale across all violated hazards
        needs_projection = False
        scale = 1.0
        for (hx, hy, hr) in self.hazards:
            d = np.linalg.norm(next_pos - np.array([hx, hy], dtype=np.float32))
            if d <= hr:
                needs_projection = True
                # Bisection: find the largest t in [0, 1] such that
                #   ||pos + predict(t * a_xy) - hazard_center|| >= hr - epsilon
                # 20 iterations gives precision ~1e-6, which is more than enough
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    test_next = pos + self._predict_displacement(mid * a_xy, heading)
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
            self.last_deflection_magnitude = float(np.linalg.norm(a_proj[:2] - a[:2]))
            return a_proj

        return action
