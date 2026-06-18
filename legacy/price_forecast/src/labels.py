import numpy as np

def build_targets(mid: np.ndarray, horizons: list[int], mode="direction", eps: float = 0.0):
    T = len(mid)
    targets = {}
    for h in horizons:
        y = np.full(T, np.nan, dtype=float)
        for t in range(T - h):
            now, fut = mid[t], mid[t + h]
            if mode == "direction":
                if eps > 0:
                    if fut > now + eps: y[t] = 1.0
                    elif fut < now - eps: y[t] = 0.0
                    else: y[t] = np.nan
                else:
                    y[t] = 1.0 if fut > now else 0.0
            elif mode == "return":
                y[t] = np.log(fut) - np.log(now)
            else:
                raise ValueError("mode must be direction or return")
        targets[h] = y


    return targets
