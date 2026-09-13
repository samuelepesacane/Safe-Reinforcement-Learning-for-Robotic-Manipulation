"""
Unit tests for the geometric and Riemannian safety shields.

Verifies that when a proposed action would move the agent inside a hazard disc,
the shield projects the action so the predicted next position stays just outside
(GenericKeepoutShield), and that the Riemannian shield's gradient deflection
points away from hazards, not toward them (D3), with position resolved from a
real environment rather than from the observation vector (D1).
"""
import unittest
import numpy as np
from src.safety.shield import GenericKeepoutShield
from src.safety.riemannian_shield import RiemannianShield


class TestGenericKeepoutShield(unittest.TestCase):
    """Tests for the 2D geometric keepout shield."""

    def test_projection_avoids_hazard(self) -> None:
        """
        The shielded action should keep the next position outside the hazard.

        The agent starts inside the hazard radius and proposes an action that
        would move it further toward the hazard center. The shield must scale
        the action so the predicted next position satisfies
        dist(next_pos, center) >= radius - epsilon.
        """
        # Single hazard centered at the origin with radius 1.0
        shield = GenericKeepoutShield(
            hazards=[(0.0, 0.0, 1.0)],
            dt=1.0,
            max_action_norm=1.0,
        )

        # Agent at (0.5, 0) moving left: without shielding it crosses the center
        obs = {"agent_pos": np.array([0.5, 0.0], dtype=np.float32)}
        action = np.array([-1.0, 0.0], dtype=np.float32)

        safe_action = shield.step(action, obs)

        next_pos = obs["agent_pos"][:2] + shield.dt * safe_action[:2]
        dist = np.linalg.norm(next_pos - np.array([0.0, 0.0], dtype=np.float32))

        # Next position must be at least (radius - epsilon) from the hazard center
        self.assertGreaterEqual(dist, 1.0 - 1e-3)
        self.assertTrue(shield.last_intervened)

    def test_heading_fit_requires_calibration(self) -> None:
        """
        kinematic_model='heading_fit' must raise at construction time if no
        body_frame_M/body_frame_b is supplied, rather than silently falling
        back to the known-wrong world_xy assumption.
        """
        with self.assertRaises(ValueError):
            GenericKeepoutShield(hazards=[(0.0, 0.0, 1.0)], kinematic_model="heading_fit")

    def test_heading_fit_requires_heading_at_step_time(self) -> None:
        """
        Even with calibration supplied, a step() call whose obs carries no
        heading must raise rather than silently using body_frame_b alone (or
        crashing on a None inside the rotation math).
        """
        shield = GenericKeepoutShield(
            hazards=[(0.0, 0.0, 1.0)],
            kinematic_model="heading_fit",
            body_frame_M=np.eye(2, dtype=np.float32),
            body_frame_b=np.zeros(2, dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            shield.step(
                np.array([1.0, 0.0], dtype=np.float32),
                {"agent_pos": np.array([0.5, 0.0], dtype=np.float32)},
            )

    def test_heading_fit_rotates_body_frame_prediction_into_world_frame(self) -> None:
        """
        Numerically pins the heading_fit math: with M=I, b=0, the predicted
        displacement is exactly R(heading) @ a_xy, not the world_xy dt*a_xy.
        A heading of +90 degrees should rotate a forward (+x) action into a
        world-frame +y displacement -- this is exactly what a fixed
        world-frame model can never produce regardless of scale, so it is
        the cleanest possible test that heading is actually being used.
        """
        shield = GenericKeepoutShield(
            hazards=[],  # no hazards: step() returns the raw action, but
            # _predict_displacement is exercised directly below regardless.
            kinematic_model="heading_fit",
            body_frame_M=np.eye(2, dtype=np.float32),
            body_frame_b=np.zeros(2, dtype=np.float32),
        )
        a_xy = np.array([1.0, 0.0], dtype=np.float32)
        heading = np.pi / 2  # facing +y in world frame
        pred = shield._predict_displacement(a_xy, heading)
        np.testing.assert_allclose(pred, np.array([0.0, 1.0]), atol=1e-5)

    def test_heading_fit_predicts_and_avoids_hazard_for_a_sideways_robot(self) -> None:
        """
        End-to-end check that heading_fit changes shield BEHAVIOR, not just
        the internal prediction: a robot facing +y (heading=90deg) whose body
        frame maps action -> (forward, lateral) = a_xy unchanged (M=I) is
        actually moving in world +y when it commands "forward" (+x action).
        Under the OLD world_xy model this same action would incorrectly
        predict movement in world +x, missing a hazard sitting to the north.
        Under heading_fit it must correctly predict the +y motion and project
        the action away from a hazard placed there.
        """
        hazard = (0.0, 1.0, 0.3)  # sits north of the robot
        pos = np.array([0.0, 0.0], dtype=np.float32)
        heading = np.pi / 2  # facing +y (i.e. "forward" is world +y)

        shield_new = GenericKeepoutShield(
            hazards=[hazard],
            dt=1.0,
            max_action_norm=1.0,
            kinematic_model="heading_fit",
            body_frame_M=np.eye(2, dtype=np.float32),
            body_frame_b=np.zeros(2, dtype=np.float32),
        )
        action = np.array([1.0, 0.0], dtype=np.float32)  # "drive forward"
        safe_action_new = shield_new.step(action, {"agent_pos": pos, "heading": heading})
        self.assertTrue(shield_new.last_intervened)

        # Old world_xy model at the same pose: "forward" is misread as world
        # +x, so it never sees the hazard sitting north and does not intervene.
        shield_old = GenericKeepoutShield(hazards=[hazard], dt=1.0, max_action_norm=1.0)
        safe_action_old = shield_old.step(action, {"agent_pos": pos, "heading": heading})
        self.assertFalse(shield_old.last_intervened)
        np.testing.assert_allclose(safe_action_old, action)

    def test_interior_override_requires_calibration(self) -> None:
        """interior_override=True must raise at construction without a fit."""
        with self.assertRaises(ValueError):
            GenericKeepoutShield(hazards=[(0.0, 0.0, 1.0)], interior_override=True)

    def test_interior_override_is_inert_outside_a_hazard(self) -> None:
        """
        Outside any hazard, interior_override must not change behavior at
        all -- it only fires once the agent's CURRENT position is already
        inside a hazard.
        """
        shield = GenericKeepoutShield(
            hazards=[(0.0, 0.0, 1.0)],
            interior_override=True,
            body_frame_M=np.eye(2, dtype=np.float32),
            body_frame_b=np.zeros(2, dtype=np.float32),
        )
        action = np.array([0.1, 0.0], dtype=np.float32)
        safe_action = shield.step(
            action, {"agent_pos": np.array([5.0, 5.0], dtype=np.float32), "heading": 0.0}
        )
        np.testing.assert_allclose(safe_action, action)
        self.assertFalse(shield.last_interior_override_fired)
        self.assertFalse(shield.last_intervened)

    def test_interior_override_replaces_inward_action_with_full_magnitude_outward_one(self) -> None:
        """
        With M=I (body frame == world frame for this test), an agent stuck
        inside a hazard proposing a strongly INWARD action must get that
        action fully replaced (not blended) with a full-magnitude
        (max_action_norm) action pointing exactly away from the hazard
        center -- this is the numeric core of the interior-override design.
        """
        hazard = (0.0, 0.0, 1.0)
        pos = np.array([0.5, 0.0], dtype=np.float32)  # inside: dist 0.5 < radius 1.0
        shield = GenericKeepoutShield(
            hazards=[hazard],
            max_action_norm=1.0,
            interior_override=True,
            body_frame_M=np.eye(2, dtype=np.float32),
            body_frame_b=np.zeros(2, dtype=np.float32),
        )
        inward_action = np.array([-1.0, 0.0], dtype=np.float32)  # straight toward the center
        safe_action = shield.step(inward_action, {"agent_pos": pos, "heading": 0.0})

        self.assertTrue(shield.last_interior_override_fired)
        self.assertTrue(shield.last_intervened)
        # Full magnitude, not alpha-scaled.
        self.assertAlmostEqual(float(np.linalg.norm(safe_action[:2])), 1.0, places=5)
        # Exactly away from the hazard center (+x direction here), not toward it.
        away_dir = np.array([1.0, 0.0], dtype=np.float32)
        cos_sim = float(np.dot(safe_action[:2], away_dir))
        self.assertGreater(cos_sim, 0.999)
        # This is a REPLACEMENT: the original inward action must play no role.
        self.assertLess(float(np.dot(safe_action[:2], inward_action)), 0.0)

    def test_interior_override_inverts_a_nontrivial_body_frame_gain(self) -> None:
        """
        With a non-identity, non-diagonal M, the override must still point
        the resulting body-frame displacement (M @ a) toward the true escape
        direction -- not just copy the escape direction into the action
        verbatim. Confirms the linear-inversion math, not just the M=I case.
        """
        hazard = (0.0, 0.0, 1.0)
        pos = np.array([0.5, 0.0], dtype=np.float32)
        heading = 0.0
        M = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)  # swaps + scales axes
        shield = GenericKeepoutShield(
            hazards=[hazard],
            max_action_norm=1.0,
            interior_override=True,
            body_frame_M=M,
            body_frame_b=np.zeros(2, dtype=np.float32),
        )
        safe_action = shield.step(
            np.array([0.0, 0.0], dtype=np.float32), {"agent_pos": pos, "heading": heading}
        )
        self.assertAlmostEqual(float(np.linalg.norm(safe_action[:2])), 1.0, places=5)
        predicted_disp = M @ safe_action[:2]  # heading=0 so body frame == world frame
        away_dir = np.array([1.0, 0.0], dtype=np.float32)
        cos_sim = float(
            np.dot(predicted_disp, away_dir) / (np.linalg.norm(predicted_disp) + 1e-8)
        )
        self.assertGreater(cos_sim, 0.999)

    def test_extract_agent_xy_raises_on_non_dict_obs(self) -> None:
        """
        D1 (second half): a broken position source must raise, not silently
        pass the action through unshielded. There is no flat-array fallback
        anymore -- Safety-Gymnasium's flat obs has no absolute position in it.
        """
        shield = GenericKeepoutShield(hazards=[(0.0, 0.0, 1.0)], dt=1.0, max_action_norm=1.0)
        with self.assertRaises(ValueError):
            shield.step(np.array([1.0, 0.0], dtype=np.float32), np.array([0.5, 0.5, 0.0]))


class TestRiemannianShieldDeflection(unittest.TestCase):
    """
    Pins D3's fix numerically: the deflection must point away from the
    hazard, with the true direction preserved (not quantized to a fixed
    diagonal by an element-wise clip on the raw gradient).
    """

    def test_deflection_points_away_from_hazard_and_preserves_direction(self) -> None:
        # Hazard at the origin, radius 1.0. Agent off-axis at (1.2, 0.3), so
        # the true away-from-hazard direction is NOT diagonal -- this is what
        # distinguishes a correct norm-based clip from the old element-wise
        # clip, which would saturate x and y independently and always point
        # at ~45 degrees regardless of the real hazard geometry.
        hazard = (0.0, 0.0, 1.0)
        pos = np.array([1.2, 0.3], dtype=np.float32)
        shield = RiemannianShield(
            hazards=[hazard],
            dt=1.0,
            max_action_norm=1.0,
            alpha=0.1,
            influence_radius=0.5,
        )

        # Zero proposed action isolates the deflection itself in safe_action.
        action = np.zeros(2, dtype=np.float32)
        safe_action = shield.step(action, {"agent_pos": pos})

        away_dir = pos - np.array([hazard[0], hazard[1]], dtype=np.float32)
        away_dir = away_dir / np.linalg.norm(away_dir)
        deflection = safe_action[:2] - action[:2]

        # Sign: must point away from the hazard, not toward it.
        self.assertGreater(float(np.dot(deflection, away_dir)), 0.0)

        # Direction: with a single hazard the true repulsive direction is
        # exactly the hazard-to-agent unit vector. Cosine similarity close to
        # 1 confirms the norm-based clip preserved it; the old element-wise
        # clip would saturate both axes to +/-max_action_norm and produce a
        # ~45-degree deflection here (cosine similarity ~0.86), which this
        # threshold would catch.
        cos_sim = float(np.dot(deflection, away_dir) / (np.linalg.norm(deflection) + 1e-8))
        self.assertGreater(cos_sim, 0.999)

        # Magnitude: gradient norm is clipped to max_action_norm before
        # scaling by alpha, so the deflection norm should be alpha * 1.0.
        self.assertAlmostEqual(float(np.linalg.norm(deflection)), 0.1, places=4)

        self.assertTrue(shield.last_intervened)

        # No bisection fallback triggers here (the deflected next position is
        # still well outside the hazard), so total deflection should equal the
        # gradient-stage deflection exactly -- and the gradient's raw norm
        # (~150) is far above max_action_norm=1.0, so the clip must have fired.
        self.assertTrue(shield.last_gradient_intervened)
        self.assertTrue(shield.last_gradient_clip_fired)
        self.assertAlmostEqual(shield.last_gradient_deflection_magnitude, 0.1, places=4)
        self.assertAlmostEqual(shield.last_deflection_magnitude, 0.1, places=4)
        self.assertAlmostEqual(
            shield.last_deflection_magnitude,
            shield.last_gradient_deflection_magnitude,
            places=6,
        )

    def test_clip_fires_even_at_the_weakest_edge_of_the_influence_zone(self) -> None:
        """
        Pins a finding from review: with max_action_norm=1.0, the raw gradient
        magnitude is |grad| = 2/clearance^3 (the 1/d factor in the code cancels
        against |diff|=d exactly). At clearance == influence_radius -- the
        weakest possible interaction, right at the outer edge of the zone --
        |grad| = 2/influence_radius^3, which for any influence_radius < ~1.26
        already exceeds max_action_norm=1.0. So the norm clip is expected to
        fire for EVERY position inside the influence zone, not just close ones,
        given the influence_radius values actually used in this project
        (0.2-0.5). That means the gradient deflection is fixed-magnitude
        (alpha * max_action_norm) whenever it fires at all, not proximity-
        proportional -- see SESSION_HANDOFF.md for the consequence for the
        shield's docstrings.
        """
        influence_radius = 0.4
        hazard_radius = 0.2
        # Just inside the zone, not exactly at the boundary: at clearance ==
        # influence_radius exactly, float32 round-off can push the computed
        # clearance a hair above influence_radius and trip the code's own
        # (correct) "clearance > influence_radius -> skip" boundary check.
        clearance = influence_radius * 0.999
        hazard = (0.0, 0.0, hazard_radius)
        pos = np.array([hazard_radius + clearance, 0.0], dtype=np.float32)

        shield = RiemannianShield(
            hazards=[hazard],
            dt=1.0,
            max_action_norm=1.0,
            alpha=0.1,
            influence_radius=influence_radius,
        )

        raw_grad = shield._barrier_gradient(pos)
        expected_raw_norm = 2.0 / (clearance ** 3)
        self.assertAlmostEqual(float(np.linalg.norm(raw_grad)), expected_raw_norm, places=2)
        self.assertGreater(float(np.linalg.norm(raw_grad)), shield.max_action_norm)

        action = np.zeros(2, dtype=np.float32)
        shield.step(action, {"agent_pos": pos})
        self.assertTrue(shield.last_gradient_clip_fired)


class TestShieldEnvIntegration(unittest.TestCase):
    """
    Integration tests against a real Safety-Gymnasium environment, pinning
    D1 (position must come from env.unwrapped, and must track motion) and
    D2 (hazards must refresh every reset) end-to-end through the actual
    wrapper wiring, not just the isolated helper functions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import safety_gymnasium  # noqa: F401
        except Exception:
            raise unittest.SkipTest("safety_gymnasium not installed")

    def test_resolve_agent_pos_matches_ground_truth_and_tracks_motion(self) -> None:
        from src.envs.make_env import _try_make, resolve_agent_pos

        env = _try_make("SafetyPointGoal1-v0", seed=0)
        env.reset(seed=0)
        uw = env.unwrapped

        pos0 = resolve_agent_pos(env)
        ground_truth0 = np.array(uw.task.agent.pos[:2], dtype=np.float32)
        np.testing.assert_allclose(pos0, ground_truth0, atol=1e-6)

        for _ in range(20):
            env.step(np.ones(env.action_space.shape, dtype=np.float32))

        pos1 = resolve_agent_pos(env)
        ground_truth1 = np.array(uw.task.agent.pos[:2], dtype=np.float32)
        np.testing.assert_allclose(pos1, ground_truth1, atol=1e-6)

        # Not just "matches ground truth" but "the ground truth itself moved" --
        # an accessor that exists but is constant would pass the equality
        # check above while still reproducing D1 somewhere new.
        self.assertGreater(float(np.linalg.norm(pos1 - pos0)), 1e-3)
        env.close()

    def test_hazards_refresh_on_wrapper_reset(self) -> None:
        from src.envs.make_env import _try_make, ShieldingActionWrapper

        env = _try_make("SafetyCarGoal1-v0", seed=0)
        shield = GenericKeepoutShield(hazards=[], dt=0.1, max_action_norm=1.0)
        wrapped = ShieldingActionWrapper(env, shield)

        wrapped.reset(seed=0)
        hazards_first = list(shield.hazards)
        self.assertTrue(len(hazards_first) > 0)

        wrapped.reset(seed=1)
        hazards_second = list(shield.hazards)
        self.assertTrue(len(hazards_second) > 0)

        # D2's whole premise: a fresh reset must load a different layout, not
        # keep serving the one captured at construction/first reset.
        self.assertNotEqual(hazards_first, hazards_second)
        env.close()


if __name__ == "__main__":
    unittest.main()
