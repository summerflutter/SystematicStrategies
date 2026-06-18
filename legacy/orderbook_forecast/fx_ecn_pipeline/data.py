import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['send_time','recv_time']).sort_values('recv_time').reset_index(drop=True)
    return df

def compute_latency_stats(df: pd.DataFrame) -> pd.DataFrame:
    dd = df.copy()
    dd['latency_s'] = (dd['recv_time'] - dd['send_time']).dt.total_seconds()
    stats = dd.groupby('venue')['latency_s'].agg(['count','mean','std','median','min','max']).reset_index()
    return stats
