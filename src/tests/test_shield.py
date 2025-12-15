"""
Unit tests for the GenericKeepoutShield geometric safety filter.

The goal is to ensure that:
- when the agent's proposed action would move it inside a hazard disc,
  the shield projects the action so that the predicted next position stays
  (slightly) outside the hazard.
"""

import unittest
import numpy as np
from src.safety.shield import GenericKeepoutShield


class TestGenericKeepoutShield(unittest.TestCase):
    """Tests for the 2D geometric keep-out shield."""

    def test_projection_avoids_hazard(self) -> None:
        """
        If the raw action would move the agent into the hazard,
        the shield should scale the action so the next position remains
        just outside the hazard radius.
        """
        # Single hazard centered at the origin with radius 1.0
        shield = GenericKeepoutShield(
            hazards=[(0.0, 0.0, 1.0)],
            dt=1.0,
            max_action_norm=1.0,
        )

        # Agent starts at x=0.5, moving left with velocity -1.0
        # Without shielding, it would cross the hazard center at (0, 0)
        obs = {"agent_pos": np.array([0.5, 0.0], dtype=np.float32)}
        action = np.array([-1.0, 0.0], dtype=np.float32)

        safe_action = shield.step(action, obs)

        # Predicted next position under the shielded action
        next_pos = obs["agent_pos"][:2] + shield.dt * safe_action[:2]

        # Distance to hazard center must be >= radius - epsilon
        center = np.array([0.0, 0.0], dtype=np.float32)
        dist = np.linalg.norm(next_pos - center)

        self.assertGreaterEqual(dist, 1.0 - 1e-3)
        # Check that the shield actually intervened
        self.assertTrue(shield.last_intervened)


if __name__ == "__main__":
    unittest.main()
