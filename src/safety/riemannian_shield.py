"""
Simplified Riemannian shield inspired by Jaquier et al. (2023).

This module implements a geometric action shield whose correction mechanism
is derived from the region-avoiding Riemannian metric idea in:

    Klein, Jaquier, Meixner, Asfour. "On the Design of Region-Avoiding Metrics
    for Collision-Safe Motion Generation on Riemannian Manifolds."
    IROS 2023. arXiv:2307.15440

In the full paper, robot motion is generated as a geodesic under a modified
metric G(q) = G_base + sum_i w_i(q) * v_i v_i^T, where w_i is a barrier
weight that grows to infinity near hazard i and v_i is the unit vector pointing
toward that hazard. Paths that would pass through a hazard become metrically
expensive, so geodesics naturally curve away from forbidden regions.

Computing true geodesics under this metric requires solving a second-order ODE
at each step, which is too expensive for a runtime RL shield. This module
implements the first-order approximation: at each step, we compute the
repulsive direction of the barrier potential (the negative gradient, which
points away from hazards) at the current position and add it to the proposed
action.

The barrier weight uses an inverse-square form:
    w_i(pos) = max(0, 1 / (d_i - r_i)^2)
where d_i is the Euclidean distance from pos to hazard center i and r_i is its
radius. The gradient of this scalar field with respect to pos is the deflection
signal, and its magnitude is |grad_i| = 2/clearance^3 (the 1/d factor in the
code's formula cancels exactly against the diff vector's own magnitude d).

IMPORTANT, found in review after D3's sign/clip fix landed: this magnitude
formula means the deflection is NOT proximity-proportional in practice, for
any influence_radius actually used in this project (0.2-0.5). At the weakest
possible interaction -- clearance == influence_radius, right at the outer edge
of the zone -- |grad| is already 16-250x max_action_norm=1.0 (see the table in
SESSION_HANDOFF.md). Since the norm clip (see RiemannianShield.step) is
applied to this raw gradient BEFORE alpha scales it, alpha cannot change
whether the clip fires -- only clearance vs max_action_norm can, and clearance
never gets large enough not to trigger it at these settings. Empirically, the
clip fires in 100% of gradient interventions on both SafetyPointGoal1-v0 and
SafetyCarGoal1-v0 (measured over a 5k-step probe at alpha=0.1,
influence_radius=0.4). So in practice this shield is a smoothly-varying-
DIRECTION, fixed-MAGNITUDE (alpha*max_action_norm) correction inside
influence_radius, and exactly zero outside it -- a soft trigger boundary with
a hard-capped push, not a force that grows continuously with proximity the way
the paragraph above (and the class docstring) describe the underlying barrier
potential's mathematical form. The direction remains correct and continuous;
only the magnitude does not vary with proximity in the operating range used
here. Getting genuine proximity-modulation would require either clipping the
alpha-scaled deflection instead of the raw gradient, or an alpha small enough
that alpha*max(|grad| over the zone) <= max_action_norm -- for influence_radius
in {0.2, 0.3, 0.4, 0.5} that threshold alpha is {0.004, 0.0135, 0.032, 0.0625}
respectively, well below the values used in this project's runs, and small
enough that the resulting deflection ceiling (a few percent of the action
range) may be too weak to matter.

The key qualitative difference from GenericKeepoutShield is that this shield
triggers earlier (at influence_radius beyond the hazard boundary, rather than
only when the predicted next position would actually enter the hazard) and,
within that trigger zone, corrects direction using the true barrier geometry
rather than GenericKeepoutShield's isotropic scale-down. It does not, in
practice, intervene "more gently" -- see above.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
from .shield import GenericKeepoutShield


class RiemannianShield(GenericKeepoutShield):
    """
    Gradient-based action deflection shield inspired by region-avoiding
    Riemannian metrics (Jaquier et al., IROS 2023).

    Instead of projecting the action by bisecting its scale when the predicted
    next position would enter a hazard, this shield computes the repulsive
    direction (negative gradient) of a barrier potential field at the current
    position and adds it to the proposed action. The barrier potential is the
    sum of inverse-square terms centered at each hazard, and its gradient's
    DIRECTION does follow the true barrier geometry continuously as the robot
    moves. Its MAGNITUDE does not, in practice: see the module docstring --
    at the influence_radius values used in this project (0.2-0.5), the raw
    gradient's norm exceeds max_action_norm even at the outer edge of the
    zone, so the norm clip saturates on essentially every intervention
    (measured: 100% of gradient interventions in a 5k-step probe on both
    robots). The result is a fixed-magnitude (alpha*max_action_norm),
    correctly-directed correction inside influence_radius and nothing outside
    it, not a force that grows with proximity.

    This is a first-order approximation of the geodesic deflection that would
    arise under Jaquier et al.'s modified metric. It is computationally cheap
    (one gradient evaluation per step). Whether an individual step's
    correction is "smooth" relative to GenericKeepoutShield's hard truncation
    depends on what varies: direction varies continuously here; magnitude, in
    the saturated regime this project runs in, does not.

    The fallback to the parent class bisection is retained for cases where the
    gradient deflection alone is insufficient: if after gradient deflection the
    predicted next position still violates a hazard, the bisection projection
    is applied as a safety net.

    :param hazards: List of hazard discs, each as (x, y, radius).
        :type hazards: Optional[List[Tuple[float, float, float]]]
    :param dt: Time step for next-position prediction.
        :type dt: float
    :param max_action_norm: Maximum allowed L2 norm of the XY action component.
        :type max_action_norm: float
    :param epsilon: Safety margin kept between predicted position and boundary.
        :type epsilon: float
    :param alpha: Scaling coefficient for the barrier gradient deflection.
        Because the norm clip is applied to the raw gradient BEFORE alpha
        scales it (see step()), alpha does not change whether the clip fires
        -- only clearance vs max_action_norm does, and at the influence_radius
        values used in this project it fires ~100% of the time regardless of
        alpha. In that saturated regime alpha is simply the deflection
        magnitude ceiling (alpha*max_action_norm), not a proximity-sensitivity
        knob.
        :type alpha: float
    :param influence_radius: Only hazards closer than this distance (beyond
        their radius) contribute to the gradient. Acts as a soft cutoff so
        distant hazards do not affect the action at all. Empirically the
        dominant driver of intervention RATE (see SESSION_HANDOFF.md's
        influence_radius scan) -- alpha and influence_radius affect different
        things: alpha sets how hard a triggered correction pushes (capped, per
        the note above), influence_radius sets how often it triggers at all.
        :type influence_radius: float
    """

    def __init__(
        self,
        hazards: Optional[List[Tuple[float, float, float]]] = None,
        dt: float = 0.1,
        max_action_norm: float = 1.0,
        epsilon: float = 1e-3,
        alpha: float = 0.1,
        influence_radius: float = 0.5,
        kinematic_model: str = "world_xy",
        body_frame_M: Optional[np.ndarray] = None,
        body_frame_b: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            hazards=hazards,
            dt=dt,
            max_action_norm=max_action_norm,
            epsilon=epsilon,
            kinematic_model=kinematic_model,
            body_frame_M=body_frame_M,
            body_frame_b=body_frame_b,
        )
        self.alpha = alpha
        # Only hazards within this extra clearance beyond their radius
        # contribute gradient, so the shield is quiet far from obstacles
        self.influence_radius = influence_radius

        # Diagnostics specific to the gradient stage, separate from the
        # inherited last_deflection_magnitude (which measures the TOTAL
        # action change across both the gradient stage and any bisection
        # fallback -- conflating the two was a bug: with max_action_norm=1.0
        # this shield's raw gradient is |grad|=2/clearance^3, which already
        # exceeds max_action_norm at clearance=influence_radius for every
        # influence_radius below ~1.26, so the norm clip saturates almost
        # every intervention and last_gradient_deflection_magnitude sits at
        # exactly alpha*max_action_norm most of the time. See
        # last_gradient_clip_fired for whether this step was one of them.
        self.last_gradient_deflection_magnitude: float = 0.0
        self.last_gradient_clip_fired: bool = False
        # True iff the gradient stage itself intervened this step (grad_norm >
        # 1e-8), independent of whether the bisection fallback also fired.
        # Lets a caller compute the clip-fire fraction CONDITIONAL on a
        # gradient intervention having happened, rather than diluted by all
        # the steps -- including bisection-only ones -- where it did not.
        self.last_gradient_intervened: bool = False

    def on_reset(self) -> None:
        """Reset episode-level state, including the Riemannian-specific diagnostics."""
        super().on_reset()
        self.last_gradient_deflection_magnitude = 0.0
        self.last_gradient_clip_fired = False
        self.last_gradient_intervened = False

    def _barrier_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier potential at the current position.

        The potential is the sum of inverse-square barriers, one per hazard:
            phi(pos) = sum_i max(0, 1 / (d_i - r_i)^2)
        where d_i = ||pos - h_i||_2 is the distance to hazard center i and
        r_i is its radius. Its true gradient with respect to pos is
            grad phi_i = -2 / (d_i - r_i)^3 * (pos - h_i) / d_i
        which points TOWARD the hazard center (phi increases as you approach
        it, so its steepest-ascent direction points inward). The safe,
        repulsive direction is the negative of that:
            -grad phi_i = 2 / (d_i - r_i)^3 * (pos - h_i) / d_i
        which is what this method returns -- it points away from the hazard
        center and grows as the boundary is approached. Adding it to the
        action (not subtracting) deflects motion away from the hazard.

        Only hazards within influence_radius beyond their boundary contribute,
        to avoid global perturbation from distant hazards.

        :param pos: Current 2D agent position [x, y].
            :type pos: np.ndarray

        :return: 2D gradient vector of the barrier potential at pos.
            :rtype: np.ndarray
        """
        grad = np.zeros(2, dtype=np.float32)

        for (hx, hy, hr) in self.hazards:
            diff = pos - np.array([hx, hy], dtype=np.float32)
            d = float(np.linalg.norm(diff))

            # Clearance: distance outside the hazard boundary
            clearance = d - hr

            # Only contribute if within influence radius and outside boundary.
            # Inside the boundary (clearance <= 0) the gradient would point
            # inward, which is wrong — the bisection fallback handles that case.
            if clearance <= 0.0 or clearance > self.influence_radius:
                continue

            # True gradient of 1/clearance^2 w.r.t. pos is
            #   -2/clearance^3 * (pos - h)/d  (points toward the hazard).
            # grad_i below is its negation, i.e. the repulsive direction:
            #   +2/clearance^3 * (pos - h)/d  (points away from the hazard).
            grad_i = (2.0 / (clearance ** 3 * (d + 1e-8))) * diff
            grad += grad_i

        return grad

    def step(self, action: np.ndarray, obs: Any) -> np.ndarray:
        """
        Apply the Riemannian-inspired barrier gradient deflection to the action.

        First computes the repulsive barrier direction at the current position
        and adds alpha * (norm-clipped repulsive direction) to the XY action
        components. Direction varies continuously with position; magnitude
        does not, in the regime this project runs in -- the norm clip
        saturates on ~100% of gradient interventions at the influence_radius
        values used here (see the module docstring), so the deflection is a
        fixed-magnitude (alpha*max_action_norm) push in the correct direction
        rather than a force that grows with proximity.

        If the deflected action still predicts a next position inside a hazard
        (which can happen very close to boundaries), the parent class bisection
        is applied as a hard safety fallback.

        :param action: Proposed action from the policy.
            :type action: np.ndarray
        :param obs: Current environment observation for position extraction.
            :type obs: Any

        :return: Deflected safe action.
            :rtype: np.ndarray
        """
        self.last_intervened = False
        self.last_gradient_deflection_magnitude = 0.0
        self.last_gradient_clip_fired = False
        self.last_gradient_intervened = False

        if not self.hazards:
            return action

        pos = self._extract_agent_xy(obs)

        a = np.array(action, dtype=np.float32)
        if a.shape[0] < 2:
            return action

        # Compute barrier gradient and deflect the action
        grad = self._barrier_gradient(pos)
        grad_norm = np.linalg.norm(grad)

        if grad_norm > 1e-8:
            # grad already points away from the hazard (see _barrier_gradient);
            # add it, scaled by alpha, so the action is deflected away rather
            # than toward the hazard (this was D3's sign error).
            #
            # Clip by norm, not element-wise: the element-wise np.clip used to
            # be applied to the raw gradient (which scales as 1/clearance^3
            # and is therefore almost always far above max_action_norm), so it
            # saturated on nearly every step and quantized the correction to a
            # fixed-magnitude, 45-degree-aligned nudge regardless of true
            # proximity or direction. Clipping the norm instead caps the
            # magnitude while preserving the true direction.
            #
            # NOTE: this clip is on the RAW gradient, before alpha scales it,
            # so whether it fires does not depend on alpha at all -- only on
            # clearance vs max_action_norm. See last_gradient_clip_fired.
            self.last_gradient_clip_fired = bool(grad_norm > self.max_action_norm)
            if self.last_gradient_clip_fired:
                grad = grad * (self.max_action_norm / grad_norm)
            deflection = self.alpha * grad
            a_deflected = a.copy()
            a_deflected[:2] = a[:2] + deflection
            self.last_intervened = True
            self.last_gradient_intervened = True
            self.interventions_in_episode += 1
            # Gradient-stage magnitude only, bounded by alpha*max_action_norm --
            # NOT the total action change (see last_deflection_magnitude below,
            # which also includes any bisection fallback and was previously
            # the only magnitude reported, conflating the two mechanisms).
            self.last_gradient_deflection_magnitude = float(np.linalg.norm(deflection))
        else:
            a_deflected = a

        # Fallback: if the deflected action still violates a hazard, apply
        # the parent bisection. This handles the edge case of being very close
        # to a boundary where gradient deflection alone is insufficient.
        #
        # super().step() unconditionally resets last_intervened on entry, so
        # capture this method's own gradient-stage flag first and OR it back
        # in afterward -- otherwise a pure gradient deflection that doesn't
        # additionally trip the bisection fallback would silently read as
        # last_intervened=False, undercounting shield_intervention_rate.
        gradient_intervened = self.last_intervened
        a_safe = super().step(a_deflected, obs)
        self.last_intervened = self.last_intervened or gradient_intervened

        # Measure total deflection from the ORIGINAL proposed action across
        # both stages combined (gradient step + any bisection fallback), not
        # from the intermediate a_deflected that super().step() measures
        # against internally.
        self.last_deflection_magnitude = float(np.linalg.norm(a_safe[:2] - a[:2]))

        return a_safe
