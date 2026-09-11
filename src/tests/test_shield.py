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
        self.assertAlmostEqual(shield.last_deflection_magnitude, 0.1, places=4)


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
