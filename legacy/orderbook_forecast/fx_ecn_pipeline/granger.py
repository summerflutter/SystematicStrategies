import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def granger_against_agg(grid_df: pd.DataFrame, venues, maxlag:int=5) -> dict:
    results = {}
    for v in venues:
        cols = [f'mid_{v}', 'agg_mid']
        if not all(c in grid_df.columns for c in cols):
            results[v] = {'note': 'missing columns'}
            continue
        pair = grid_df[cols].dropna().astype(float)
        if pair.shape[0] < 20:
            results[v] = {'note':'too few samples'}
            continue
        try:
            test = grangercausalitytests(pair[['agg_mid', f'mid_{v}']], maxlag=maxlag, verbose=False)
            pvals = {lag: float(test[lag][0]['ssr_ftest'][1]) for lag in test}
            results[v] = pvals
        except Exception as e:
            results[v] = {'error': str(e)}
    return results
