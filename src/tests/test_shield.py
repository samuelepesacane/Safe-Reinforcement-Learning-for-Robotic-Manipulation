"""
Unit tests for the GenericKeepoutShield geometric safety filter.

Verifies that when a proposed action would move the agent inside a hazard disc,
the shield projects the action so the predicted next position stays just outside.
"""
import unittest
import numpy as np
from src.safety.shield import GenericKeepoutShield


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


if __name__ == "__main__":
    unittest.main()
