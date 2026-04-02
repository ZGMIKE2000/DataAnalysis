"""
Data loader for IMC Prosperity market replay.
Handles CSV discovery, parsing, normalization, and joining of prices + trades data.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def discover_data_dirs(root: str) -> list[dict]:
    """Walk root directory and find folders containing Prosperity price/trade CSVs.
    Returns list of dicts with keys: path, season, round, label."""
    results = []
    root = Path(root)

    for csv_path in sorted(root.rglob("prices_round_*.csv")):
        folder = csv_path.parent
        if any(r["path"] == str(folder) for r in results):
            continue

        m = re.search(r"prices_round_(\d+)_day_", csv_path.name)
        if not m:
            continue
        round_num = int(m.group(1))

        # Detect season from the closest path component, not the full path
        # (avoids false matches when the repo itself sits inside "Prosperity 4/")
        parts = [p.lower() for p in folder.parts]
        if "prosperity4" in parts:
            season = 4
        elif "prosperity3" in parts:
            season = 3
        else:
            # Fallback: check folder name fragments
            folder_lower = folder.name.lower()
            if "prosperity4" in folder_lower or "prosperity 4" in folder_lower:
                season = 4
            elif "prosperity3" in folder_lower or "prosperity 3" in folder_lower:
                season = 3
            else:
                season = 0

        label = f"S{season} Round {round_num} — {folder.name}"
        results.append({
            "path": str(folder),
            "season": season,
            "round": round_num,
            "label": label,
        })

    return results


def _read_csv_robust(filepath: str) -> pd.DataFrame:
    """Read a semicolon-delimited Prosperity CSV, handling common issues."""
    try:
        df = pd.read_csv(filepath, sep=";", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(filepath, sep=";", error_bad_lines=False)
    return df


def load_prices(folder: str) -> pd.DataFrame:
    """Load and concatenate all prices CSVs in a folder."""
    files = sorted(glob.glob(os.path.join(folder, "prices_round_*.csv")))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = _read_csv_robust(f)
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices.columns = prices.columns.str.strip().str.lower()

    if "product" in prices.columns:
        prices.rename(columns={"product": "instrument"}, inplace=True)

    numeric_cols = [c for c in prices.columns if c not in ("instrument",)]
    for col in numeric_cols:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    prices["instrument"] = prices["instrument"].astype(str).str.strip().str.upper()

    # Composite key for ordering (always monotonic even across negative days)
    if "day" in prices.columns and "timestamp" in prices.columns:
        prices["ts_key"] = prices["day"] * 1_000_000 + prices["timestamp"]
    elif "timestamp" in prices.columns:
        prices["ts_key"] = prices["timestamp"]

    prices.sort_values(["instrument", "ts_key"], inplace=True)
    prices.reset_index(drop=True, inplace=True)

    # Build per-instrument sequential step index (always 0, 1, 2, ...)
    prices["step_idx"] = prices.groupby("instrument").cumcount()

    return prices


def load_trades(folder: str) -> pd.DataFrame:
    """Load and concatenate all trades CSVs in a folder."""
    files = sorted(glob.glob(os.path.join(folder, "trades_round_*.csv")))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = _read_csv_robust(f)
        m = re.search(r"day_([-\d]+)", f)
        if m:
            df["day"] = int(m.group(1))
        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)
    trades.columns = trades.columns.str.strip().str.lower()

    if "symbol" in trades.columns:
        trades.rename(columns={"symbol": "instrument"}, inplace=True)

    trades["instrument"] = trades["instrument"].astype(str).str.strip().str.upper()

    numeric_cols = ["timestamp", "price", "quantity", "day"]
    for col in numeric_cols:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    if "day" in trades.columns and "timestamp" in trades.columns:
        trades["ts_key"] = trades["day"] * 1_000_000 + trades["timestamp"]
    elif "timestamp" in trades.columns:
        trades["ts_key"] = trades["timestamp"]

    trades.sort_values(["instrument", "ts_key"], inplace=True)
    trades.reset_index(drop=True, inplace=True)
    return trades


def load_round_data(folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both prices and trades for a folder. Returns (prices_df, trades_df)."""
    return load_prices(folder), load_trades(folder)


def build_instrument_index(prices: pd.DataFrame, instrument: str) -> dict:
    """Pre-build fast lookup structures for a single instrument.
    Returns dict with:
        inst_prices: filtered+sorted DataFrame
        ts_keys:     numpy array of ts_key values
        step_idxs:   numpy array of step_idx values
        ts_to_step:  dict mapping ts_key -> step_idx
        step_to_ts:  dict mapping step_idx -> ts_key
        day_boundaries: list of step_idx where day changes
        labels:      dict mapping step_idx -> display label string
    """
    mask = prices["instrument"] == instrument
    inst = prices.loc[mask].copy()
    if inst.empty:
        return None

    ts_keys = inst["ts_key"].values
    step_idxs = inst["step_idx"].values

    ts_to_step = dict(zip(ts_keys, step_idxs))
    step_to_ts = dict(zip(step_idxs, ts_keys))

    # Build display labels and find day boundaries
    labels = {}
    day_boundaries = []
    prev_day = None
    for _, row in inst.iterrows():
        s = int(row["step_idx"])
        day = int(row["day"]) if "day" in row.index and pd.notna(row["day"]) else None
        ts = int(row["timestamp"]) if "timestamp" in row.index and pd.notna(row["timestamp"]) else None

        if day is not None and ts is not None:
            labels[s] = f"t={ts}"
        elif ts is not None:
            labels[s] = f"t={ts}"
        else:
            labels[s] = str(s)

        if day is not None and day != prev_day:
            day_boundaries.append({"step_idx": s, "day": day})
            prev_day = day

    return {
        "inst_prices": inst,
        "ts_keys": ts_keys,
        "step_idxs": step_idxs,
        "ts_to_step": ts_to_step,
        "step_to_ts": step_to_ts,
        "day_boundaries": day_boundaries,
        "labels": labels,
    }


def get_instruments(prices: pd.DataFrame) -> list[str]:
    """Return sorted list of unique instruments."""
    if prices.empty or "instrument" not in prices.columns:
        return []
    return sorted(prices["instrument"].dropna().unique().tolist())


def get_step_count(prices: pd.DataFrame, instrument: str) -> int:
    """Return total number of timestamp steps for an instrument."""
    mask = prices["instrument"] == instrument
    return int(mask.sum())


def get_orderbook_at_step(prices: pd.DataFrame, instrument: str, step_idx: int) -> dict:
    """Extract order book snapshot at a specific step index.
    Returns dict with 'bids' and 'asks' lists of (price, volume) tuples, plus mid_price."""
    mask = (prices["instrument"] == instrument) & (prices["step_idx"] == step_idx)
    row = prices.loc[mask]
    if row.empty:
        return {"bids": [], "asks": [], "mid_price": None, "day": None, "timestamp": None}

    row = row.iloc[0]
    bids = []
    asks = []
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            bids.append((float(bp), int(bv)))
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and av > 0:
            asks.append((float(ap), int(av)))

    mid = row.get("mid_price")
    day = row.get("day")
    ts = row.get("timestamp")

    return {
        "bids": bids,
        "asks": asks,
        "mid_price": float(mid) if pd.notna(mid) else None,
        "day": int(day) if pd.notna(day) else None,
        "timestamp": int(ts) if pd.notna(ts) else None,
    }


def get_trades_at_step(
    trades: pd.DataFrame, instrument: str, ts_key: int
) -> pd.DataFrame:
    """Return all trades for an instrument at a given ts_key."""
    if trades.empty:
        return pd.DataFrame()
    mask = (trades["instrument"] == instrument) & (trades["ts_key"] == ts_key)
    return trades.loc[mask].copy()


def get_price_context_by_step(
    prices: pd.DataFrame, instrument: str, step_idx: int, window: int = 50
) -> pd.DataFrame:
    """Return price rows for an instrument in a window of steps around step_idx."""
    mask = prices["instrument"] == instrument
    inst = prices.loc[mask]
    if inst.empty:
        return pd.DataFrame()

    lo = max(0, step_idx - window)
    hi = step_idx + window + 1
    ctx_mask = (inst["step_idx"] >= lo) & (inst["step_idx"] < hi)
    return inst.loc[ctx_mask].copy()


def get_trades_in_step_range(
    trades: pd.DataFrame,
    instrument: str,
    ts_key_lo: int,
    ts_key_hi: int,
) -> pd.DataFrame:
    """Return trades for an instrument within a ts_key range."""
    if trades.empty:
        return pd.DataFrame()
    mask = (
        (trades["instrument"] == instrument)
        & (trades["ts_key"] >= ts_key_lo)
        & (trades["ts_key"] <= ts_key_hi)
    )
    return trades.loc[mask].copy()


def format_step_label(day, timestamp) -> str:
    """Format a single step as a readable label."""
    if day is not None and timestamp is not None:
        return f"Day {day} | t={timestamp}"
    if timestamp is not None:
        return f"t={timestamp}"
    return "—"
