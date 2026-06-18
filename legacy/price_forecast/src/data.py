# src/data.py
import glob, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence, Literal

@dataclass
class LOBData:
    bids_px: np.ndarray  # [T, L]
    bids_sz: np.ndarray  # [T, L]
    asks_px: np.ndarray  # [T, L]
    asks_sz: np.ndarray  # [T, L]
    ts: Optional[np.ndarray] = None      # [T]
    group: Optional[np.ndarray] = None   # [T] group id: file or date string

def _stack_list_col(series: pd.Series, L: int, pad_value: float, dtype=float) -> np.ndarray:
    T = len(series)
    out = np.full((T, L), pad_value, dtype=dtype)
    for i, x in enumerate(series.values):
        if x is None:
            continue
        arr = np.asarray(x, dtype=dtype)
        n = min(len(arr), L)
        if n:
            out[i, :n] = arr[:n]
    return out

def load_lob_data_nested_multi(
    paths: Sequence[str] | str,
    L: int,
    col_ts: str = "ts",
    col_bid_px: str = "bid_prices",
    col_bid_sz: str = "bid_sizes",
    col_ask_px: str = "ask_prices",
    col_ask_sz: str = "ask_sizes",
    sort_by_time: bool = True,
    drop_bad_rows: bool = True,
    group_by: Literal["file","date","none"] = "file",
    tz: str = "UTC",
) -> LOBData:
    # Resolve file list
    if isinstance(paths, str):
        if os.path.isdir(paths):
            file_list = sorted(glob.glob(os.path.join(paths, "*.parquet")))
        else:
            file_list = sorted(glob.glob(paths))
    else:
        file_list = list(paths)
    if not file_list:
        raise ValueError("No parquet files found for given paths.")

    use_cols = [col_bid_px, col_bid_sz, col_ask_px, col_ask_sz]
    if col_ts is not None:
        use_cols.append(col_ts)

    dfs = []
    for fp in file_list:
        df = pd.read_parquet(fp, columns=use_cols)
        df["_file"] = os.path.basename(fp)
        dfs.append(df)

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # Sort globally by ts (important when files overlap)
    ts = None
    if col_ts in df_all.columns and sort_by_time:
        df_all = df_all.sort_values(col_ts, kind="mergesort").reset_index(drop=True)
        ts = df_all[col_ts].to_numpy()
    elif col_ts in df_all.columns:
        ts = df_all[col_ts].to_numpy()

    # group key
    group = None
    if group_by == "file":
        group = df_all["_file"].astype(str).to_numpy()
    elif group_by == "date":
        if ts is None:
            raise ValueError("group_by='date' requires a timestamp column.")
        # best-effort: handle numeric/str/datetime
        ts_dt = pd.to_datetime(df_all[col_ts], utc=True, errors="coerce")
        # convert to date string like '2025-12-24'
        group = ts_dt.dt.tz_convert(tz).dt.date.astype(str).to_numpy()
    elif group_by == "none":
        group = None
    else:
        raise ValueError("group_by must be file/date/none")

    # Integrity filtering (len match + non-empty)
    if drop_bad_rows:
        good = []
        for bp, bs, ap, a_s in zip(df_all[col_bid_px].values, df_all[col_bid_sz].values,
                                  df_all[col_ask_px].values, df_all[col_ask_sz].values):
            if bp is None or bs is None or ap is None or a_s is None:
                good.append(False); continue
            good.append((len(bp)==len(bs)) and (len(ap)==len(a_s)) and (len(bp)>0) and (len(ap)>0))
        good = np.asarray(good, dtype=bool)
        df_all = df_all.loc[good].reset_index(drop=True)
        if ts is not None: ts = ts[good]
        if group is not None: group = group[good]

    bids_px = _stack_list_col(df_all[col_bid_px], L=L, pad_value=np.nan, dtype=float)
    bids_sz = _stack_list_col(df_all[col_bid_sz], L=L, pad_value=0.0, dtype=float)
    asks_px = _stack_list_col(df_all[col_ask_px], L=L, pad_value=np.nan, dtype=float)
    asks_sz = _stack_list_col(df_all[col_ask_sz], L=L, pad_value=0.0, dtype=float)

    # Drop bad best bid/ask
    if drop_bad_rows:
        best_bid = bids_px[:, 0]
        best_ask = asks_px[:, 0]
        mask = np.isfinite(best_bid) & np.isfinite(best_ask) & (best_bid > 0) & (best_ask > 0)
        bids_px, bids_sz = bids_px[mask], bids_sz[mask]
        asks_px, asks_sz = asks_px[mask], asks_sz[mask]
        if ts is not None: ts = ts[mask]
        if group is not None: group = group[mask]

    return LOBData(bids_px=bids_px, bids_sz=bids_sz, asks_px=asks_px, asks_sz=asks_sz, ts=ts, group=group)
