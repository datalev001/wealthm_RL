"""
replay_buffer.py
----------------
Disk-efficient replay buffer with variable-length trajectories.
Returns fixed-length mini-batches + attention masks.

API
---
add(traj_dict)          : append one trajectory (np.ndarray fields)
save(path)              : persist as .npz
load(path) -> buffer    : class-method constructor
sample(B, T) -> tuple   : (S, A, R, G, mask)
                           shapes:
                             S  (B, T, dim_s)   float32
                             A  (B, T, dim_a)   float32
                             R  (B, T)          float32
                             G  (B, T, 3)       float32
                             M  (B, T)          float32  (1=valid)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Trajectory:
    """Single customer trajectory."""
    states:  np.ndarray   # (T, dim_s) float32
    actions: np.ndarray   # (T, dim_a) float32
    rewards: np.ndarray   # (T,)        float32
    g_vecs:  np.ndarray   # (T, 3)      float32


@dataclass
class ReplayBuffer:
    trajectories: List[Trajectory] = field(default_factory=list)

    # ---------- writer side ---------- #
    def add(self, traj_dict: dict[str, np.ndarray]):
        """traj_dict keys: states, actions, rewards, g_vecs."""
        self.trajectories.append(Trajectory(**traj_dict))

    def save(self, path: str | Path):
        np.savez_compressed(
            path,
            **{
                f"traj_{i}": np.asarray(
                    [t.states, t.actions, t.rewards, t.g_vecs], dtype=object
                )
                for i, t in enumerate(self.trajectories)
            },
        )

    # ---------- reader side ---------- #
    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        data = np.load(path, allow_pickle=True)
        rb = cls()
        for key in sorted(data.files):
            states, actions, rewards, g = data[key]
            rb.add(
                dict(
                    states=states.astype(np.float32, copy=False),
                    actions=actions.astype(np.float32, copy=False),
                    rewards=rewards.astype(np.float32, copy=False),
                    g_vecs=g.astype(np.float32, copy=False),
                )
            )
        return rb

    # ---------- sampling ---------- #
    def sample(self, batch_size: int, horizon: int, rng=np.random):
        """
        Random batch with temporal slices.  Zero-pad shorter segments and
        return an attention mask (1 for valid, 0 for pad).
        """
        idx_sel = rng.choice(len(self.trajectories), batch_size, replace=True)

        S, A, R, G, M = [], [], [], [], []
        for i in idx_sel:
            traj = self.trajectories[i]
            T = len(traj.rewards)

            start = 0 if T <= horizon else rng.integers(0, T - horizon)
            end   = min(start + horizon, T)

            s = traj.states[start:end]
            a = traj.actions[start:end]
            r = traj.rewards[start:end]
            g = traj.g_vecs[start:end]

            pad = horizon - (end - start)
            if pad:
                s = np.pad(s, ((0, pad), (0, 0)), mode="constant", constant_values=0)
                a = np.pad(a, ((0, pad), (0, 0)), mode="constant", constant_values=0)
                r = np.pad(r, (0, pad), mode="constant", constant_values=0)
                g = np.pad(g, ((0, pad), (0, 0)), mode="constant", constant_values=0)

            mask = np.zeros(horizon, dtype=np.float32)
            mask[: horizon - pad] = 1.0

            S.append(s); A.append(a); R.append(r); G.append(g); M.append(mask)

        return (
            np.stack(S).astype(np.float32),
            np.stack(A).astype(np.float32),
            np.stack(R).astype(np.float32),
            np.stack(G).astype(np.float32),
            np.stack(M).astype(np.float32),
        )

    # ---------- python helpers ---------- #
    def __len__(self):
        return len(self.trajectories)