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
implements the first-order approximation: at each step, we compute the gradient
of the barrier potential at the current position and subtract it from the
proposed action. This deflects the action away from hazards in proportion to
proximity, reproducing the qualitative behavior of the metric-modified geodesic
without the full geodesic computation.

The barrier weight uses an inverse-square form:
    w_i(pos) = max(0, 1 / (d_i - r_i)^2)
where d_i is the Euclidean distance from pos to hazard center i and r_i is its
radius. The gradient of this scalar field with respect to pos is the deflection
signal. This is a smooth, differentiable barrier that grows continuously to
infinity as the robot approaches the hazard boundary, unlike the hard truncation
in the bisection-based GenericKeepoutShield.

The key qualitative difference from GenericKeepoutShield is that this shield
produces a continuous gradient-based deflection that acts at a distance, rather
than a binary projection that only activates when the next predicted position
would enter the hazard. This means the shield intervenes more gently and more
often, which is closer in spirit to how a metric-modified geodesic behaves.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
from .shield import GenericKeepoutShield


class RiemannianShield(GenericKeepoutShield):
    """
    Gradient-based action deflection shield inspired by region-avoiding
    Riemannian metrics (Jaquier et al., IROS 2023).

    Instead of projecting the action by bisecting its scale when the predicted
    next position would enter a hazard, this shield computes the gradient of a
    barrier potential field at the current position and subtracts it from the
    proposed action. The barrier potential is the sum of inverse-square terms
    centered at each hazard, so the deflection grows smoothly as the robot
    approaches any hazard boundary.

    This is a first-order approximation of the geodesic deflection that would
    arise under Jaquier et al.'s modified metric. It is computationally cheap
    (one gradient evaluation per step) and produces smooth, continuous
    interventions rather than hard truncations.

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
        Larger values produce stronger deflection at a given distance.
        :type alpha: float
    :param influence_radius: Only hazards closer than this distance (beyond
        their radius) contribute to the gradient. Acts as a soft cutoff so
        distant hazards do not affect the action at all.
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
    ) -> None:
        super().__init__(
            hazards=hazards,
            dt=dt,
            max_action_norm=max_action_norm,
            epsilon=epsilon,
        )
        self.alpha = alpha
        # Only hazards within this extra clearance beyond their radius
        # contribute gradient, so the shield is quiet far from obstacles
        self.influence_radius = influence_radius

    def _barrier_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the barrier potential at the current position.

        The potential is the sum of inverse-square barriers, one per hazard:
            phi(pos) = sum_i max(0, 1 / (d_i - r_i)^2)
        where d_i = ||pos - h_i||_2 is the distance to hazard center i and
        r_i is its radius. The gradient of each term with respect to pos is:
            grad phi_i = -2 / (d_i - r_i)^3 * (pos - h_i) / d_i
        which points away from the hazard center and grows as the boundary
        is approached. This is the direction of steepest ascent of the
        potential, i.e. the direction that most increases the barrier cost.
        Subtracting it from the action deflects motion away from the hazard.

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

            # Gradient of 1/clearance^2 with respect to pos:
            #   d/d_pos [1/clearance^2] = -2/clearance^3 * d(clearance)/d_pos
            #   d(clearance)/d_pos = (pos - h) / d
            # Combined: -2 / (clearance^3 * d) * (pos - h)
            # We negate to get the repulsive direction (away from hazard)
            grad_i = (2.0 / (clearance ** 3 * (d + 1e-8))) * diff
            grad += grad_i

        return grad

    def step(self, action: np.ndarray, obs: Any) -> np.ndarray:
        """
        Apply the Riemannian-inspired barrier gradient deflection to the action.

        First computes the barrier gradient at the current position and
        subtracts alpha * gradient from the XY action components. This
        deflects the action away from nearby hazards continuously and
        proportionally to proximity, approximating the deflection that would
        arise under Jaquier et al.'s modified metric.

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

        if not self.hazards:
            return action

        pos = self._extract_agent_xy(obs)
        if pos is None:
            return action

        a = np.array(action, dtype=np.float32)
        if a.shape[0] < 2:
            return action

        # Compute barrier gradient and deflect the action
        grad = self._barrier_gradient(pos)
        grad_norm = np.linalg.norm(grad)

        if grad_norm > 1e-8:
            # Subtract the gradient, scaled by alpha and clipped to max_action_norm
            # so the deflection cannot produce an arbitrarily large correction
            deflection = self.alpha * np.clip(
                grad, -self.max_action_norm, self.max_action_norm
            )
            a_deflected = a.copy()
            a_deflected[:2] = a[:2] - deflection
            self.last_intervened = True
            self.interventions_in_episode += 1
        else:
            a_deflected = a

        # Fallback: if the deflected action still violates a hazard, apply
        # the parent bisection. This handles the edge case of being very close
        # to a boundary where gradient deflection alone is insufficient.
        a_safe = super().step(a_deflected, obs)

        # If the parent did not intervene further, last_intervened reflects
        # only the gradient step. If it did, both flags are set.
        return a_safe
