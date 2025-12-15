from typing import Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PreferenceReward(nn.Module):
    """
    Neural reward model used for preference-based learning.

    This model maps a flattened observation vector to a scalar reward estimate
    :math:`\\hat{r}(s)`. It is implemented as a small multi-layer perceptron (MLP).

    :param input_dim: Dimensionality of the flattened observation vector.
    :type input_dim: int
    :param hidden_sizes: Sizes of the hidden layers in the MLP.
    :type hidden_sizes: List[int]
    """

    def __init__(self, input_dim: int, hidden_sizes: List[int] = [128, 128]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the reward model.

        :param x: Batch of flattened observations, shape ``(batch_size, input_dim)``.
        :type x: torch.Tensor

        :return: Predicted scalar rewards, shape ``(batch_size,)``.
        :rtype: torch.Tensor
        """
        return self.net(x).squeeze(-1)


def flatten_obs(obs: Any) -> np.ndarray:
    """
    Flatten a single observation to a 1D float32 NumPy array.

    If the observation is a dict, all entries are converted to arrays and
    concatenated in sorted key order. Otherwise, the observation is directly
    converted to a NumPy array and flattened.

    :param obs: Observation from the environment (array-like or dict).
    :type obs: Any

    :return: Flattened observation vector.
    :rtype: numpy.ndarray
    """
    if isinstance(obs, dict):
        parts: List[np.ndarray] = []
        keys = sorted(obs.keys())
        for k in keys:
            v = obs[k]
            arr = np.array(v, dtype=np.float32).ravel()
            parts.append(arr)
        return np.concatenate(parts, axis=0).astype(np.float32)
    else:
        return np.array(obs, dtype=np.float32).ravel()


def segment_score(
    model: PreferenceReward,
    segment: List[Tuple[Any, Any, float]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the model's total reward estimate for a trajectory segment.

    The segment is a list of (obs, action, reward) triples. Only the observations
    are used; the environment reward is ignored here because the goal is to learn
    a reward model that could, in principle, differ from the environment reward.

    :param model: Reward model to evaluate.
    :type model: PreferenceReward
    :param segment: Sequence of (obs, action, reward) triples.
    :type segment: List[Tuple[Any, Any, float]]
    :param device: Device on which to run the model ("cpu" or "cuda").
    :type device: str

    :return: Scalar tensor containing the sum of model-predicted rewards
        over the segment.
    :rtype: torch.Tensor
    """
    obs_array = np.stack([flatten_obs(s[0]) for s in segment], axis=0)
    x_t = torch.from_numpy(obs_array).to(device)
    scores = model(x_t)
    return scores.sum()


def train_preference_model(
    model: PreferenceReward,
    pairs: List[Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> PreferenceReward:
    """
    Train a PreferenceReward model from pairwise segment preferences.

    Each element in ``pairs`` is a tuple ``(seg_a, seg_b, pref)`` where:
      - ``seg_a`` and ``seg_b`` are trajectory segments (lists of (obs, action, reward))
      - ``pref = 0`` means segment A is preferred,
      - ``pref = 1`` means segment B is preferred.

    The model is trained using a Bradley-Terry style objective:
      - predict scores ``s_a = sum r_hat(s_t)`` and ``s_b = sum r_hat(s_t)``,
      - use ``logit = s_a - s_b`` as input to BCEWithLogitsLoss,
      - label is 1 if A is preferred, 0 if B is preferred.

    :param model: Reward model to train.
    :type model: PreferenceReward
    :param pairs: List of (segment_A, segment_B, preference) triples.
    :type pairs: List[Tuple[List[Tuple[Any, Any, float]], List[Tuple[Any, Any, float]], int]]
    :param epochs: Number of training epochs to run.
    :type epochs: int
    :param batch_size: Batch size for preference updates.
    :type batch_size: int
    :param lr: Learning rate for the Adam optimizer.
    :type lr: float
    :param device: Device on which to run training ("cpu" or "cuda").
    :type device: str

    :return: The trained reward model (same instance as input).
    :rtype: PreferenceReward
    """
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    data = pairs
    n = len(data)

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0.0

        for i in range(0, n, batch_size):
            batch = data[i : i + batch_size]
            if not batch:
                continue

            labels: List[float] = []
            logits: List[torch.Tensor] = []

            for seg_a, seg_b, pref in batch:
                # pref == 0 -> A preferred, pref == 1 -> B preferred
                sa = segment_score(model, seg_a, device=device)
                sb = segment_score(model, seg_b, device=device)

                # logit for "A is preferred"
                logit = sa - sb
                logits.append(logit)
                labels.append(1.0 if pref == 0 else 0.0)

            if not logits:
                continue

            logits_t = torch.stack(logits)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            loss = loss_fn(logits_t, labels_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        # print(f"[Pref model] epoch={epoch} loss={total_loss / max(1, (n // batch_size)):.4f}")

    return model
