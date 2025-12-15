from typing import List, Tuple, Any
import numpy as np
import random


class TrajectoryBuffer:
    """
    Buffer for storing trajectories and generating synthetic preference data.

    Trajectories are stored as lists of (obs, action, reward) triples. The buffer
    can be used to:
      - extract fixed-length segments from stored trajectories, and
      - generate pairwise comparisons between segments using the *oracle* reward
        (i.e. the environment's ground-truth reward) via a Bradley-Terry model.

    This is useful for simulating human preferences in a controlled way, so that
    preference-based reward models can be trained and evaluated without requiring
    real human feedback.

    :param segment_len: Length (in timesteps) of each segment used for comparisons.
    :type segment_len: int
    :param seed: Random seed used for sampling segments and preferences.
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

        A trajectory is a list of (obs, action, reward) tuples, where `obs` and
        `action` can be arbitrary Python objects (typically NumPy arrays) and
        `reward` is a scalar float.

        :param traj: Trajectory to add to the buffer.
        :type traj: List[Tuple[Any, Any, float]]
        """
        self.trajectories.append(traj)

    def sample_segments(
        self,
        n_pairs: int,
        noise: float = 0.1,
    ) -> List[Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]]:
        """
        Sample pairs of segments with synthetic preferences.

        Steps:
          1. Extract all contiguous segments of length ``segment_len`` from each
             stored trajectory.
          2. Uniformly sample two segments A and B from this pool.
          3. Compute their *oracle returns* ``R(A) = sum r_t`` and
             ``R(B) = sum r_t``.
          4. Use a Bradley-Terry preference model

                 p(A preferred over B) = sigma(R(A) - R(B))

             and optionally mix with uniform noise to generate a binary label.

        The returned list contains tuples of the form:

            (segment_A, segment_B, choice)

        where ``choice = 0`` means A is preferred, and ``choice = 1`` means B
        is preferred.

        :param n_pairs: Number of segment pairs to sample.
        :type n_pairs: int
        :param noise: Amount of uniform noise to mix into the Bradley-Terry
            preference probability. ``noise=0`` means pure Bradley-Terry;
            ``noise=1`` means completely random preferences.
        :type noise: float

        :return: List of (segment_A, segment_B, choice) tuples.
        :rtype: List[Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]]
        """
        pairs: List[
            Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]
        ] = []
        segments: List[List[Tuple[Any, Any, float]]] = []

        # Extract fixed-length segments from all trajectories
        for traj in self.trajectories:
            if len(traj) >= self.segment_len:
                for i in range(0, len(traj) - self.segment_len + 1, self.segment_len):
                    seg = traj[i : i + self.segment_len]
                    segments.append(seg)

        if len(segments) < 2:
            # Not enough data to form a pair
            return pairs

        for _ in range(n_pairs):
            a, b = random.sample(segments, 2)
            ra = sum(step[2] for step in a)
            rb = sum(step[2] for step in b)

            # Bradley-Terry with optional noise
            pa = 1.0 / (1.0 + np.exp(-(ra - rb)))
            pa = pa * (1.0 - noise) + 0.5 * noise

            # choice = 0 -> A preferred, choice = 1 -> B preferred
            choice = 0 if random.random() < pa else 1
            pairs.append((a, b, choice))

        return pairs
