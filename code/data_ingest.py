import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 1 · Configuration
# --------------------------------------------------------------------------- #
RAW_DIR = Path(r"C:/backupcgi/final_bak")
FILE_MAP = {
    "static":   RAW_DIR / "static_profiles.csv",
    "flows":    RAW_DIR / "dynamic_flows.csv",
    "assets":   RAW_DIR / "asset_liability.csv",
    "market":   RAW_DIR / "market_data.csv",
    "actions":  RAW_DIR / "action_execution_log.csv",
    "feedback": RAW_DIR / "constraint_feedback.csv",
}
DATE_COLS = {
    "flows":    "date",
    "market":   "date",
    "actions":  "date",
    "feedback": "date",
}

# derive master month grid from market_data
MARKET = pd.read_csv(FILE_MAP["market"], parse_dates=[DATE_COLS["market"]])
MASTER_DATES = pd.date_range(
    MARKET[DATE_COLS["market"]].min(),
    MARKET[DATE_COLS["market"]].max(),
    freq="MS",
)

# --------------------------------------------------------------------------- #
# 2 · Helper functions
# --------------------------------------------------------------------------- #
def _load_with_date(fp: Path, date_col: str | None) -> pd.DataFrame:
    """Read CSV, optionally parsing *date_col* into Timestamp."""
    if date_col:
        return pd.read_csv(fp, parse_dates=[date_col])
    return pd.read_csv(fp)


def _assert_columns(df: pd.DataFrame, cols: List[str], table: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")


def _monthly_reindex(
    df: pd.DataFrame, date_col: str, zero_fill: bool = False
) -> pd.DataFrame:
    """
    Re-index so that every (customer, month) pair exists.
    Numerical columns are forward-filled; optionally zero-filled
    (required for cash-flow style tables).
    """
    df = df.copy()
    df.set_index(["customer_id", date_col], inplace=True)
    df = df.sort_index()

    full_idx = pd.MultiIndex.from_product(
        [df.index.get_level_values(0).unique(), MASTER_DATES],
        names=["customer_id", date_col],
    )
    df = df.reindex(full_idx)

    num_cols = df.select_dtypes("number").columns
    if zero_fill:
        df[num_cols] = df[num_cols].fillna(0.0)
    else:
        df[num_cols] = df[num_cols].groupby("customer_id").ffill()
    return df.reset_index()


def _expand_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Cartesian product snapshot → (customer_id, date) grid."""
    df = df.copy()
    df["key"] = 1
    return (
        df.merge(pd.DataFrame({"date": MASTER_DATES, "key": 1}), on="key", how="outer")
        .drop(columns="key")
        .sort_values(["customer_id", "date"])
    )


# --------------------------------------------------------------------------- #
# 3 · Main ETL routine
# --------------------------------------------------------------------------- #
def build_master_hdf(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "master_timeseries.h5"

    print("↳ loading CSV files …")
    static   = _load_with_date(FILE_MAP["static"],   None)
    flows    = _load_with_date(FILE_MAP["flows"],    DATE_COLS["flows"])
    assets   = _load_with_date(FILE_MAP["assets"],   None)
    actions  = _load_with_date(FILE_MAP["actions"],  DATE_COLS["actions"])
    feedback = _load_with_date(FILE_MAP["feedback"], DATE_COLS["feedback"])

    # basic sanity
    _assert_columns(static,  ["customer_id"],           "static")
    _assert_columns(flows,   ["customer_id", "date"],   "flows")
    _assert_columns(assets,  ["customer_id"],           "assets")
    _assert_columns(actions, ["customer_id", "date"],   "actions")
    _assert_columns(feedback,["customer_id", "date"],   "feedback")

    # month-align tables that already have a date
    flows    = _monthly_reindex(flows,   "date", zero_fill=True)
    actions  = _monthly_reindex(actions, "date", zero_fill=False)
    feedback = _monthly_reindex(feedback,"date", zero_fill=False)

    # expand snapshot-style tables
    static_exp  = _expand_snapshot(static)
    assets_exp  = _expand_snapshot(assets)

    print("↳ writing master_timeseries.h5 …")
    with pd.HDFStore(h5_path, mode="w") as store:
        store.put("static",   static_exp,  format="table", data_columns=True)
        store.put("flows",    flows,       format="table", data_columns=True)
        store.put("assets",   assets_exp,  format="table", data_columns=True)
        store.put("actions",  actions,     format="table", data_columns=True)
        store.put("feedback", feedback,    format="table", data_columns=True)
        store.put("market",   MARKET,      format="table", data_columns=True)

    # ------------------------------------------------------------------ #
    # 4 · QA checks
    # ------------------------------------------------------------------ #
    print("↳ running QA checks …")

    # A. completeness of flows grid
    assert (
        flows.groupby("customer_id")["date"].nunique() == len(MASTER_DATES)
    ).all(), "Flows table has missing months after re-index."

    # B. execution ≈ recommendation × compliance sanity
    if {"rec_delta_equity", "exec_delta_equity", "compliance_rate"}.issubset(
        actions.columns
    ):
        actions["calc_exec_eq"] = (
            actions["rec_delta_equity"] * actions["compliance_rate"]
        )
        err = np.abs(actions["calc_exec_eq"] - actions["exec_delta_equity"])
        assert (
            err.median() < 1e-3
        ), "Median absolute error between executed and implied equity deltas > 1e-3"

    print(f"✓ Master file saved → {h5_path}")


# --------------------------------------------------------------------------- #
# 5 · CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=r"C:/backupcgi/final_bak/processed",
        help="Directory to save processed HDF5 store",
    )
    args = parser.parse_args()
    build_master_hdf(Path(args.out_dir))

