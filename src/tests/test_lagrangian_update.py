"""
Unit tests for the LagrangianState dual update.

These tests verify that:
- the Lagrange multiplier lambda increases when the average cost exceeds
  the target budget, and
- lambda is never driven below zero (projection onto [0, +inf)).
"""

import unittest
from src.algos.lagppo import LagrangianState


class TestLagrangianStateUpdate(unittest.TestCase):
    """Tests for the LagrangianState.update dual ascent step."""

    def test_lambda_increases_when_cost_above_budget(self) -> None:
        """
        If the observed average cost is above the budget, lambda should increase.

        This corresponds to tightening the constraint in a CMDP dual update.
        """
        st = LagrangianState(lam=0.0, lr_lambda=1e-3, cost_budget=0.05)
        lam1 = st.update(avg_cost_per_step=0.10)
        self.assertGreater(lam1, 0.0)

    def test_lambda_clamped_at_zero(self) -> None:
        """
        If the update would push lambda below zero, it should be clamped at 0.

        This ensures the Lagrange multiplier remains in the feasible region
        lambda greater or equal than 0.
        """
        st = LagrangianState(lam=0.1, lr_lambda=1e-1, cost_budget=0.10)
        lam1 = st.update(avg_cost_per_step=0.0)
        self.assertGreaterEqual(lam1, 0.0)


if __name__ == "__main__":
    unittest.main()
