# src/split.py
import numpy as np

def group_time_split_indices(
    group: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    group: [T] array, e.g. file name per row or date string per row.
    We split by ordered unique groups (preserving temporal order).
    """
    if group is None:
        raise ValueError("group_time_split_indices requires group array (file/date).")

    # get ordered unique groups by first occurrence
    _, first_idx = np.unique(group, return_index=True)
    ordered_groups = group[np.sort(first_idx)]

    G = len(ordered_groups)
    if G < 3:
        raise ValueError(f"Need at least 3 groups for train/val/test, got {G}.")

    n_train = max(1, int(G * train_ratio))
    n_val = max(1, int(G * val_ratio))
    # ensure at least 1 group in test
    if n_train + n_val >= G:
        n_val = max(1, G - n_train - 1)

    train_groups = set(ordered_groups[:n_train])
    val_groups = set(ordered_groups[n_train:n_train+n_val])
    test_groups = set(ordered_groups[n_train+n_val:])

    train_idx = np.where(np.isin(group, list(train_groups)))[0]
    val_idx   = np.where(np.isin(group, list(val_groups)))[0]
    test_idx  = np.where(np.isin(group, list(test_groups)))[0]

    return train_idx, val_idx, test_idx

def time_split_indices(T: int, train_ratio: float=0.7, val_ratio: float=0.15):
    n_train = int(T*train_ratio)
    n_val = int(T*val_ratio)
    idx = np.arange(T)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
