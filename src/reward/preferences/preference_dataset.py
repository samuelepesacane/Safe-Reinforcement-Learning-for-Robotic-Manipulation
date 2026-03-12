"""
Trajectory buffer and synthetic preference generation for preference-based RL.

Supports the --use_preferences condition in the ablation grid. This condition
was explored during the study but is not included in the paper's main results.
"""
from typing import List, Tuple, Any
import numpy as np
import random


class TrajectoryBuffer:
    """
    Buffer for storing trajectories and generating synthetic preference data.

    Trajectories are stored as lists of (obs, action, reward) triples. The
    buffer extracts fixed-length segments from stored trajectories and generates
    pairwise comparisons using the oracle reward via a Bradley-Terry model.

    This simulates human preferences in a controlled way so that preference-based
    reward models can be trained and evaluated without real human feedback.

    :param segment_len: Length in timesteps of each segment used for comparisons.
        :type segment_len: int
    :param seed: Random seed for sampling segments and preferences.
        :type seed: int
    """

    def __init__(self, segment_len: int = 25, seed: int = 0) -> None:
        self.segment_len: int = segment_len
        self.trajectories: List[List[Tuple[Any, Any, float]]] = []
        random.seed(seed)
        np.random.seed(seed)

    def add_trajectory(self, traj: List[Tuple[Any, Any, float]]) -> None:
        """
        Add a new trajectory to the buffer.

        A trajectory is a list of (obs, action, reward) tuples where obs and
        action are typically NumPy arrays and reward is a scalar float.

        :param traj: Trajectory to add.
            :type traj: List[Tuple[Any, Any, float]]

        :return: None.
            :rtype: None
        """
        self.trajectories.append(traj)

    def sample_segments(
        self,
        n_pairs: int,
        noise: float = 0.1,
    ) -> List[Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]]:
        """
        Sample pairs of segments with synthetic Bradley-Terry preferences.

        Extracts all contiguous segments of length segment_len from stored
        trajectories, randomly samples pairs, computes oracle returns, and
        assigns a binary preference label using the Bradley-Terry model with
        optional uniform noise:

            p(A preferred over B) = sigma(R(A) - R(B)) * (1 - noise) + 0.5 * noise

        choice=0 means A is preferred; choice=1 means B is preferred.

        :param n_pairs: Number of segment pairs to sample.
            :type n_pairs: int
        :param noise: Fraction of uniform noise mixed into the preference
            probability. 0 means pure Bradley-Terry; 1 means random labels.
            :type noise: float

        :return: List of (segment_A, segment_B, choice) tuples.
            :rtype: List[Tuple[...]]
        """
        pairs: List[
            Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]
        ] = []
        segments: List[List[Tuple[Any, Any, float]]] = []

        for traj in self.trajectories:
            if len(traj) >= self.segment_len:
                for i in range(0, len(traj) - self.segment_len + 1, self.segment_len):
                    segments.append(traj[i : i + self.segment_len])

        # Need at least two segments to form a comparison pair
        if len(segments) < 2:
            return pairs

        for _ in range(n_pairs):
            a, b = random.sample(segments, 2)
            ra = sum(step[2] for step in a)
            rb = sum(step[2] for step in b)

            # Bradley-Terry probability with noise mixing
            pa = 1.0 / (1.0 + np.exp(-(ra - rb)))
            pa = pa * (1.0 - noise) + 0.5 * noise

            choice = 0 if random.random() < pa else 1
            pairs.append((a, b, choice))

        return pairs
