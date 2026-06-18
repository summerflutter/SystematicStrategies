import pandas as pd

def _steps(delta_ms:int, grid_ms:int)->int:
    return max(1, int(delta_ms / grid_ms))

def build_classification_target(df: pd.DataFrame, grid_ms:int, delta_ms:int, thr:float=0.0) -> pd.DataFrame:
    dd = df.copy().sort_values('grid_time').reset_index(drop=True)
    steps = _steps(delta_ms, grid_ms)
    future = dd['agg_mid'].shift(-steps)
    dd['target_dir'] = (future > dd['agg_mid'] + thr).astype(int)
    return dd

def build_regression_target(df: pd.DataFrame, grid_ms:int, delta_ms:int) -> pd.DataFrame:
    dd = df.copy().sort_values('grid_time').reset_index(drop=True)
    steps = _steps(delta_ms, grid_ms)
    dd['future_agg'] = dd['agg_mid'].shift(-steps)
    dd['target_delta'] = dd['future_agg'] - dd['agg_mid']
    return dd
