# ---------------------------------------------------
# env.py
# ---------------------------------------------------
"""
Minimal environment for personalised portfolio RL.

Key responsibilities
--------------------
1) Build state s_t from the master HDF5 store.
2) Apply an *executed* action a_exec (history or agent-supplied).
3) Update asset values using the month’s market returns.
4) Compute constraint vector g = [g_risk, g_cash, g_tax] and reward r_t.
5) Expose Gym-style reset / step APIs.

Reward
------
r_t = ΔNetWorth_after_tax − α·max(g_risk,0) − β·g_cash − γ·g_tax

NOTE  This is an **experimentation class**; production code in the
`wealth_rl` package handles concurrency, cache warm-up, and Prometheus hooks.
"""
from pathlib import Path

import numpy as np
import pandas as pd


class FinanceEnv:
    RISK_CAP = 0.15      # 1-yr VaR limit
    ALPHA, BETA, GAMMA = 1.0, 1.0, 1.0   # penalty weights

    def __init__(self, h5_path: str | Path, rho: float = 1.0):
        self.h5 = pd.HDFStore(h5_path, mode="r")
        self.rho = rho

        # snapshot tables
        self.static   = self.h5["static"].set_index(["customer_id", "date"])
        self.assets   = self.h5["assets"].set_index(["customer_id", "date"])
        self.flows    = self.h5["flows"].set_index(["customer_id", "date"])
        self.actions  = self.h5["actions"].set_index(["customer_id", "date"])
        self.feedback = self.h5["feedback"].set_index(["customer_id", "date"])
        self.market   = self.h5["market"].set_index("date")

        self.dates = self.market.index.sort_values().to_list()

        # runtime cursor
        self._cursor: tuple[int, int] | None = None   # (cid, idx)
        self._state  = None
        self._net_worth = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def reset(self, customer_id: int):
        """Reset trajectory; return initial numeric state vector."""
        self._cursor = (customer_id, 0)
        date0 = self.dates[0]
        self._net_worth = self._calc_net_worth(customer_id, date0)
        self._state = self._build_state(customer_id, date0)
        return self._state

    def step(self,
             action_exec: dict[str, float] | None = None,
             use_historical: bool = True):
        """
        Advance one month.
        Parameters
        ----------
        action_exec : dict, optional
            Agent-supplied deltas; ignored if use_historical=True.
        use_historical : bool
            *True*  → replay logged execution.
            *False* → apply action_exec supplied by caller.
        Returns
        -------
        s_next : np.ndarray
        r_t    : float
        g_vec  : np.ndarray  shape (3,)
        done   : bool
        """
        cid, i = self._cursor
        if i >= len(self.dates) - 1:
            raise RuntimeError("Trajectory finished; call reset first.")

        date_now, date_next = self.dates[i], self.dates[i + 1]

        # 1) executed action --------------------------------------------------
        if use_historical and (cid, date_now) in self.actions.index:
            row = self.actions.loc[(cid, date_now)].fillna(0.0)
            a_exec = dict(delta_equity    = row["exec_delta_equity"],
                          delta_bond      = row["exec_delta_bond"],
                          delta_gic       = row["exec_delta_gic"],
                          prepay_mortgage = row["exec_prepay_mortgage"])
        elif not use_historical:
            if action_exec is None:
                raise ValueError("action_exec must be provided when use_historical=False")
            a_exec = action_exec
        else:  # missing log → zero action
            a_exec = dict(delta_equity=0.0, delta_bond=0.0,
                          delta_gic=0.0, prepay_mortgage=0.0)

        # 2) mark-to-market ---------------------------------------------------
        asset_row = self.assets.loc[(cid, date_now)].copy()
        mkt_row   = self.market.loc[date_now]

        equity_val = asset_row.equity_value * (1 + mkt_row["equity_mom_ret"]) \
                     + a_exec["delta_equity"]
        bond_val   = asset_row.bond_value   * (1 + mkt_row["bond_mom_ret"])   \
                     + a_exec["delta_bond"]
        gic_val    = asset_row.gic_value    * (1 + mkt_row["gic_rate_pct"] / 1200) \
                     + a_exec["delta_gic"]
        mort_bal   = max(asset_row.mortgage_balance - a_exec["prepay_mortgage"], 0.0)

        # persist forward for continuity
        self.assets.loc[(cid, date_next),
                        ["equity_value", "bond_value", "gic_value", "mortgage_balance"]] = \
            [equity_val, bond_val, gic_val, mort_bal]

        # 3) constraints & reward --------------------------------------------
        net_worth = equity_val + bond_val + gic_val + asset_row.insurance_cash_value - mort_bal

        fb = self.feedback.loc[(cid, date_now)]
        g_risk = fb.VaR_95 - self.RISK_CAP * net_worth
        g_cash = fb.cash_shortfall
        g_tax  = max(-fb.tax_net_worth_gap, 0.0)

        g_vec = np.array([g_risk, g_cash, g_tax], dtype=np.float32)

        growth = (net_worth - self._net_worth) / (self._net_worth + 1e-6)
        r_t = growth - (self.ALPHA * max(g_risk, 0.0) +
                        self.BETA  * g_cash +
                        self.GAMMA * g_tax)

        # 4) advance cursor ---------------------------------------------------
        self._cursor = (cid, i + 1)
        self._net_worth = net_worth
        self._state = self._build_state(cid, date_next)
        done = (i + 1 == len(self.dates) - 1)
        return self._state, float(r_t), g_vec, done

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _build_state(self, cid: int, date) -> np.ndarray:
        """Return numeric state vector (np.ndarray)."""
        s_dict = {}
        s_dict.update(self.static.loc[(cid, date)].to_dict())
        s_dict.update(self.assets.loc[(cid, date)].to_dict())
        s_dict.update(self.flows.loc[(cid, date)].to_dict())
        s_dict.update(self.market.loc[date].to_dict())

        s = pd.Series(s_dict)
        s_num = s[pd.to_numeric(s, errors="coerce").notna()].astype(float)
        return s_num.to_numpy(dtype=np.float32)

    def _calc_net_worth(self, cid: int, date) -> float:
        row = self.assets.loc[(cid, date)]
        return (
            row.equity_value +
            row.bond_value +
            row.gic_value +
            row.insurance_cash_value -
            row.mortgage_balance
        )

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def rollout(self, customer_id: int) -> pd.DataFrame:
        """Return full trajectory DataFrame for one customer (offline)."""
        records = []
        state = self.reset(customer_id)
        done = False
        while not done:
            state, r, g, done = self.step()
            date_idx = self._cursor[1] - 1  # state already advanced
            records.append(
                {
                    "date": self.dates[date_idx],
                    "reward": r,
                    "g_risk": g[0],
                    "g_cash": g[1],
                    "g_tax":  g[2],
                    "net_worth": self._net_worth,
                }
            )
        return pd.DataFrame(records)
