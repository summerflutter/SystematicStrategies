# src/eval.py
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

def evaluate_basic(y_true, y_pred, mode="direction"):
    if mode == "direction":
        return {
            "auc": float(roc_auc_score(y_true, y_pred)),
            "logloss": float(log_loss(y_true, y_pred, eps=1e-15)),
            "brier": float(brier_score_loss(y_true, y_pred)),
        }
    err = y_pred - y_true
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))}

def ht_tl_aligned(
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    mid: np.ndarray,
    anchor_idx: np.ndarray,
    h: int,
    p_up: np.ndarray,
    threshold: float = 0.55,
):
    """
    All inputs are FULL arrays (length T) except:
      - anchor_idx: the t's we are evaluating (e.g. test anchors)
      - p_up: predicted probabilities for those anchors (same length as anchor_idx)
    We compute horizon_mid using explicit t+h indexing.
    """
    future_idx = anchor_idx + h
    horizon_mid = mid[future_idx]

    bid_now = best_bid[anchor_idx]
    ask_now = best_ask[anchor_idx]

    hit = np.full(len(anchor_idx), np.nan, dtype=float)
    adverse = np.full(len(anchor_idx), np.nan, dtype=float)

    buy = p_up > threshold
    sell = p_up < (1.0 - threshold)

    # BUY bias -> adverse is down through bid
    hit[buy] = (horizon_mid[buy] < bid_now[buy]).astype(float)
    adverse[buy] = np.maximum(0.0, bid_now[buy] - horizon_mid[buy])

    # SELL bias -> adverse is up through ask
    hit[sell] = (horizon_mid[sell] > ask_now[sell]).astype(float)
    adverse[sell] = np.maximum(0.0, horizon_mid[sell] - ask_now[sell])

    valid = np.isfinite(hit)
    ht_rate = float(np.mean(hit[valid])) if valid.any() else float("nan")

    adv = adverse[np.isfinite(adverse)]
    tl_q95 = float(np.quantile(adv, 0.95)) if len(adv) else float("nan")

    return {"hit_through_rate": ht_rate, "tail_loss_q95": tl_q95}
