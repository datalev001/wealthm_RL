"""
models.py
---------
Light-weight neural modules used throughout the paper.

Architectures
-------------
Actor         : LSTM(state) → FC → tanh      → 4-D action in (-1,1)
Critic_λ      : FC(state ⊕ action) → Softplus → 3 dual multipliers  λ ≥ 0
Critic_y      : FC(state ⊕ action) → Sigmoid  → 4 compliance ratios q ∈ (0,1)

All .forward(...) methods are **fully batched** and expect tensors shaped
(B, T, D) – returning (B, T, output_dim).  The code stays intentionally
compact so the entire pipeline trains in minutes on a single RTX-3060.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# action semantics:  Δequity, Δbond, Δgic, prepay_mortgage
ACT_DIM   = 4
STATE_DIM = 60          # will be overridden at runtime once batch is known


class Actor(nn.Module):
    """Stochastic policy πθ :  s → a  (continuous deltas, tanh-scaled)."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, ACT_DIM),
            nn.Tanh(),                     # output range (-1, +1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, state_dim)
        Returns
        -------
        torch.Tensor
            Shape (B, T, ACT_DIM) in (-1,1)
        """
        h, _ = self.lstm(x)                # (B, T, hidden)
        return self.head(h)                # (B, T, 4)


class CriticLambda(nn.Module):
    """Dual network ψφ(s,a) → λ   (non-negative via Softplus)."""

    def __init__(self, state_dim: int = STATE_DIM + ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softplus(),                 # ensure λ ≥ 0
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns λ for each constraint, shape (B, T, 3).
        """
        sa = torch.cat([s, a], dim=-1)     # concat along feature dim
        return self.net(sa)


class CriticY(nn.Module):
    """Follower model χφ(s,a) → q  –  execution compliance in (0,1)."""

    def __init__(self, state_dim: int = STATE_DIM + ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, ACT_DIM),
            nn.Sigmoid(),                  # (0,1) compliance per action dim
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns q, shape (B, T, ACT_DIM).
        """
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)