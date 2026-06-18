# run_train.py (project root)
import yaml
import numpy as np

from src.data import load_lob_data_nested_multi
from src.features import build_feature_matrix, compute_mid
from src.labels import build_targets
from src.split import group_time_split_indices
from src.align import build_anchor_mask
from src.train import fit_predict_one_h
from src.eval import ht_tl_aligned

def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r"))
    L = int(cfg["lob_levels"])
    horizons = list(cfg["horizons"])
    mode = cfg["mode"]
    eps = float(cfg["eps"])
    threshold = float(cfg["threshold"])

    data = load_lob_data_nested_multi(
        paths="DATA_DIR/*.parquet",
        L=L,
        col_ts="ts",
        col_bid_px="bid_prices",
        col_bid_sz="bid_sizes",
        col_ask_px="ask_prices",
        col_ask_sz="ask_sizes",
        group_by="file",  # or "date"
        sort_by_time=True,
        drop_bad_rows=True,
        tz="UTC",
    )

    X = build_feature_matrix(data.bids_px, data.bids_sz, data.asks_px, data.asks_sz, L=L)
    best_bid = data.bids_px[:, 0]
    best_ask = data.asks_px[:, 0]
    mid = compute_mid(best_bid, best_ask)

    targets = build_targets(mid, horizons=horizons, mode=mode, eps=eps)

    # anchor indices: valid for ALL horizons
    max_h = max(horizons)
    anchor_mask = build_anchor_mask(T=len(mid), max_h=max_h)
    anchor_idx_all = np.where(anchor_mask)[0]

    # Group split on anchors only (so future t+h never leaks into test anchors)
    if data.group is None:
        raise ValueError("Need data.group for GroupTimeSplit. Set group_by='file' or 'date' in loader.")
    group_anchor = data.group[anchor_idx_all]
    tr_a, va_a, te_a = group_time_split_indices(
        group=group_anchor,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
    )
    # map back to original index space
    train_idx = anchor_idx_all[tr_a]
    val_idx   = anchor_idx_all[va_a]
    test_idx  = anchor_idx_all[te_a]

    print(f"anchors: {len(anchor_idx_all)} | train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    for h in horizons:
        print(f"\n=== Horizon h={h} ===")
        y_full = targets[h]  # length T (NaN near end; anchors avoid it already)

        out = fit_predict_one_h(
            X=X,
            y_full=y_full,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            mode=mode,
            logreg_C=cfg["model"]["logreg_C"],
            logreg_max_iter=cfg["model"]["logreg_max_iter"],
        )
        print("VAL :", out["val"]["metrics"])
        print("TEST:", out["test"]["metrics"])

        if mode == "direction":
            te_used_idx = out["test"]["idx"]      # anchors actually used in test (finite labels)
            te_pred = out["test"]["pred"]
            ht = ht_tl_aligned(
                best_bid=best_bid,
                best_ask=best_ask,
                mid=mid,
                anchor_idx=te_used_idx,
                h=h,
                p_up=te_pred,
                threshold=threshold,
            )
            print("HT/TL:", ht)

if __name__ == "__main__":
    main()
