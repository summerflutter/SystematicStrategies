# src/train.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from .eval import evaluate_basic

def fit_predict_one_h(
    X: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mode: str,
    logreg_C: float,
    logreg_max_iter: int,
):
    # Filter finite labels within each split (eps过滤时很有用)
    tr = train_idx[np.isfinite(y_full[train_idx])]
    va = val_idx[np.isfinite(y_full[val_idx])]
    te = test_idx[np.isfinite(y_full[test_idx])]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X[tr])
    Xva_s = scaler.transform(X[va])
    Xte_s = scaler.transform(X[te])

    if mode == "direction":
        model = LogisticRegression(penalty="l2", C=logreg_C, max_iter=logreg_max_iter, n_jobs=-1)
        model.fit(Xtr_s, y_full[tr])
        pva = model.predict_proba(Xva_s)[:, 1]
        pte = model.predict_proba(Xte_s)[:, 1]
    else:
        model = Ridge(alpha=1.0)
        model.fit(Xtr_s, y_full[tr])
        pva = model.predict(Xva_s)
        pte = model.predict(Xte_s)

    val_metrics = evaluate_basic(y_full[va], pva, mode)
    test_metrics = evaluate_basic(y_full[te], pte, mode)

    # IMPORTANT: return the exact indices used for eval alignment
    return {
        "model": model,
        "scaler": scaler,
        "val": {"idx": va, "y": y_full[va], "pred": pva, "metrics": val_metrics},
        "test": {"idx": te, "y": y_full[te], "pred": pte, "metrics": test_metrics},
    }
