import numpy as np

EPS = 1e-12

def compute_mid(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    return 0.5 * (best_bid + best_ask)

def compute_spread(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    return best_ask - best_bid

def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, EPS))

def lag(arr: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """Shift forward by k: out[t] = arr[t-k]."""
    out = np.full_like(arr, fill, dtype=float)
    if k <= 0:
        return arr.astype(float)
    out[k:] = arr[:-k]
    return out

def diff(arr: np.ndarray, k: int = 1, fill: float = np.nan) -> np.ndarray:
    """Difference: arr[t] - arr[t-k]."""
    return arr - lag(arr, k, fill=fill)

def ewm(arr: np.ndarray, alpha: float, init_with_first: bool = True) -> np.ndarray:
    """
    Exponentially weighted moving average (past-only).
    out[t] = alpha*arr[t] + (1-alpha)*out[t-1]
    """
    out = np.empty_like(arr, dtype=float)
    if len(arr) == 0:
        return out
    out[0] = arr[0] if init_with_first else 0.0
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t-1]
    return out

def clip_inf_nan(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    y = x.astype(float)
    y[~np.isfinite(y)] = fill
    return y

def microprice(best_bid: np.ndarray, best_ask: np.ndarray, bid_sz0: np.ndarray, ask_sz0: np.ndarray) -> np.ndarray:
    """
    Microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    (pressure-weighted mid)
    """
    denom = bid_sz0 + ask_sz0 + EPS
    return (best_ask * bid_sz0 + best_bid * ask_sz0) / denom

def build_feature_matrix(
    bids_px: np.ndarray, bids_sz: np.ndarray,
    asks_px: np.ndarray, asks_sz: np.ndarray,
    L: int,
    return_lags=(1, 2, 5, 10),
    ewm_alphas=(0.05, 0.2),
    use_level_distances=True,
    use_level_sizes=True,
    use_level_spreads=True,
) -> np.ndarray:
    """
    Inputs: [T, L] arrays (px may have NaN padding, sz padded with 0).
    Output: X [T, D]
    """
    T = bids_px.shape[0]
    L = min(L, bids_px.shape[1])

    bid0 = bids_px[:, 0]
    ask0 = asks_px[:, 0]
    mid = compute_mid(bid0, ask0)
    spr = compute_spread(bid0, ask0)

    # ---------- (A) Core top-of-book features ----------
    # normalized spread, sizes, imbalance
    bid_sz0 = bids_sz[:, 0]
    ask_sz0 = asks_sz[:, 0]

    imb1 = (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0 + EPS)   # level-1 OI
    mp = microprice(bid0, ask0, bid_sz0, ask_sz0)
    mp_minus_mid = (mp - mid) / (mid + EPS)

    # ---------- (B) Multi-level order imbalance / depth ----------
    bid_depth = np.nansum(bids_sz[:, :L], axis=1)
    ask_depth = np.nansum(asks_sz[:, :L], axis=1)
    imball = (bid_depth - ask_depth) / (bid_depth + ask_depth + EPS)

    # depth concentration: top1 share
    bid_top_share = bid_sz0 / (bid_depth + EPS)
    ask_top_share = ask_sz0 / (ask_depth + EPS)

    # depth spread imbalance: compare near vs far depth (e.g., top K vs remaining)
    K = min(3, L)  # you can tune
    bid_near = np.nansum(bids_sz[:, :K], axis=1)
    ask_near = np.nansum(asks_sz[:, :K], axis=1)
    bid_far = np.nansum(bids_sz[:, K:L], axis=1) if L > K else np.zeros(T)
    ask_far = np.nansum(asks_sz[:, K:L], axis=1) if L > K else np.zeros(T)

    near_imb = (bid_near - ask_near) / (bid_near + ask_near + EPS)
    far_imb  = (bid_far  - ask_far)  / (bid_far  + ask_far  + EPS)

    # ---------- (C) Lagged returns / volatility proxies ----------
    log_mid = safe_log(mid)
    r1 = diff(log_mid, 1)                 # 1-event log return
    abs_r1 = np.abs(r1)

    ret_feats = [r1, abs_r1]
    for k in return_lags:
        ret_feats.append(diff(log_mid, k))           # k-event return
        ret_feats.append(np.abs(diff(log_mid, k)))   # magnitude

    # EWM of returns (trend) and abs returns (vol proxy)
    ewm_feats = []
    for a in ewm_alphas:
        ewm_feats.append(ewm(r1, a))
        ewm_feats.append(ewm(abs_r1, a))

    # ---------- (D) Spread dynamics ----------
    # spread level and change
    spr_norm = spr / (mid + EPS)
    d_spr1 = diff(spr_norm, 1)

    # ---------- (E) Order level (shape) features ----------
    # We avoid raw absolute px (scale issues), use distances to mid normalized by mid.
    level_feats = []

    if use_level_distances:
        bid_dist = (bids_px[:, :L] - mid[:, None]) / (mid[:, None] + EPS)  # negative
        ask_dist = (asks_px[:, :L] - mid[:, None]) / (mid[:, None] + EPS)  # positive
        # replace NaN pads with 0 (or a constant) to keep linear model stable
        level_feats.append(np.nan_to_num(bid_dist, nan=0.0))
        level_feats.append(np.nan_to_num(ask_dist, nan=0.0))

    if use_level_sizes:
        # size profile (optionally log1p to compress tails)
        level_feats.append(np.log1p(np.nan_to_num(bids_sz[:, :L], nan=0.0)))
        level_feats.append(np.log1p(np.nan_to_num(asks_sz[:, :L], nan=0.0)))

    if use_level_spreads:
        # within-side price gaps (queue shape): bid_px[i]-bid_px[i+1], ask_px[i+1]-ask_px[i]
        if L >= 2:
            bid_gaps = bids_px[:, :L-1] - bids_px[:, 1:L]
            ask_gaps = asks_px[:, 1:L] - asks_px[:, :L-1]
            level_feats.append(np.nan_to_num(bid_gaps / (mid[:, None] + EPS), nan=0.0))
            level_feats.append(np.nan_to_num(ask_gaps / (mid[:, None] + EPS), nan=0.0))

    # slope / convexity proxies using distances & cumulative depth
    # weighted price distance by size (pressure measure per side)
    bid_pressure = np.nansum(np.nan_to_num(bids_sz[:, :L], nan=0.0) * np.nan_to_num((mid[:, None] - bids_px[:, :L])/(mid[:, None]+EPS), nan=0.0), axis=1)
    ask_pressure = np.nansum(np.nan_to_num(asks_sz[:, :L], nan=0.0) * np.nan_to_num((asks_px[:, :L] - mid[:, None])/(mid[:, None]+EPS), nan=0.0), axis=1)
    pressure_imb = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + EPS)

    # ---------- (F) Event intensity proxy (optional, ts not always available) ----------
    # If you later want: dt = ts[t]-ts[t-1], then features like 1/dt, ewm(dt), etc.

    # ---------- Assemble ----------
    scalar_blocks = [
        mid, spr_norm, d_spr1,
        bid_sz0, ask_sz0,
        imb1, imball,
        bid_top_share, ask_top_share,
        near_imb, far_imb,
        mp_minus_mid,
        pressure_imb,
    ]

    scalar_blocks += ret_feats
    scalar_blocks += ewm_feats

    X_parts = [clip_inf_nan(np.column_stack(scalar_blocks), fill=0.0)]
    for blk in level_feats:
        # blk is [T, d]
        X_parts.append(clip_inf_nan(blk, fill=0.0))

    X = np.concatenate(X_parts, axis=1)
    return X
