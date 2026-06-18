import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def hasbrouck_info_share(grid_df: pd.DataFrame, nlags:int=3) -> pd.DataFrame:
    mid_cols = [c for c in grid_df.columns if c.startswith('mid_')]
    ts = grid_df[mid_cols].dropna().astype(float)
    if ts.shape[0] < (nlags+5):
        return pd.DataFrame({'venue':[c.replace('mid_','') for c in mid_cols], 'info_share':[np.nan]*len(mid_cols)})
    ts_log = np.log(ts)
    model = VAR(ts_log)
    try:
        res = model.fit(nlags)
        fevd = res.fevd(5)
        decomp_h = fevd.decomp[0]
        contrib = decomp_h.sum(axis=0)
        info_share = contrib / contrib.sum()
        info_df = pd.DataFrame({'venue':[c.replace('mid_','') for c in mid_cols], 'info_share': info_share})
    except Exception:
        info_df = pd.DataFrame({'venue':[c.replace('mid_','') for c in mid_cols], 'info_share':[np.nan]*len(mid_cols)})
    return info_df
