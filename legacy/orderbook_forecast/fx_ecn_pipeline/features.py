import pandas as pd

def build_features_from_grid(grid_df: pd.DataFrame, venues) -> pd.DataFrame:
    dd = grid_df.copy()
    for v in venues:
        mid_col = f'mid_{v}'; depth_col = f'depth_{v}'
        if mid_col not in dd.columns: 
            continue
        dd[mid_col] = dd[mid_col].astype(float)
        dd[depth_col] = dd[depth_col].astype(float)
        dd[f'{mid_col}_ret_1'] = dd[mid_col].pct_change(1)
        dd[f'{mid_col}_ret_3'] = dd[mid_col].pct_change(3)
        dd[f'{mid_col}_volatility_10'] = dd[mid_col].rolling(10, min_periods=1).std()
        dd[f'{mid_col}_diff_to_agg'] = dd[mid_col] - dd['agg_mid']
    depth_cols = [f'depth_{v}' for v in venues if f'depth_{v}' in dd.columns]
    dd['total_depth'] = dd[depth_cols].sum(axis=1)
    for v in venues:
        if f'depth_{v}' in dd.columns:
            dd[f'depth_imbalance_{v}'] = (dd[f'depth_{v}'] - (dd['total_depth'] - dd[f'depth_{v}'])) / (dd['total_depth'] + 1e-9)
    return dd
