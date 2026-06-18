import pandas as pd

def build_calendar_grid_merge_asof(df: pd.DataFrame, grid_ms: int = 100) -> pd.DataFrame:
    dd = df.copy()
    start = dd['recv_time'].min().floor('S')
    end = dd['recv_time'].max().ceil('S')
    grid = pd.date_range(start=start, end=end, freq=f'{grid_ms}ms')
    grid_df = pd.DataFrame({'grid_time': grid})
    venues = sorted(dd['venue'].unique())
    merged_parts = [grid_df]
    for v in venues:
        sub = dd[dd['venue']==v].sort_values('recv_time')[['recv_time','mid_price','depth']]
        sub = sub.rename(columns={'mid_price': f'mid_{v}', 'depth': f'depth_{v}'})
        tmp = pd.merge_asof(grid_df, sub, left_on='grid_time', right_on='recv_time', direction='backward')
        tmp = tmp.drop(columns=['recv_time'])
        merged_parts.append(tmp[[f'mid_{v}', f'depth_{v}']])
    combined = pd.concat(merged_parts, axis=1)
    mid_cols = [c for c in combined.columns if c.startswith('mid_')]
    depth_cols = [c for c in combined.columns if c.startswith('depth_')]
    combined['agg_mid'] = (combined[mid_cols].fillna(method='ffill') * combined[depth_cols].fillna(method='ffill')).sum(axis=1) / (combined[depth_cols].fillna(method='ffill').sum(axis=1) + 1e-12)
    return combined
