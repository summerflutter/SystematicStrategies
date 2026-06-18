# src/align.py
import numpy as np

def build_anchor_mask(T: int, max_h: int) -> np.ndarray:
    # valid anchors: t <= T - max_h - 1
    return np.arange(T) < (T - max_h)

def horizon_future_index(anchor_idx: np.ndarray, h: int) -> np.ndarray:
    return anchor_idx + h
