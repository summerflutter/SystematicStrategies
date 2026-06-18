import numpy as np
import pandas as pd

def backtest_threshold_strategy(df_probs: pd.DataFrame, 
                                price_col: str = 'agg_mid', 
                                prob_col: str = 'prob_up',
                                entry_th: float = 0.55,
                                exit_th: float = 0.5,
                                fee_bps: float = 0.1,
                                slip_bps: float = 0.05) -> pd.DataFrame:
    """Simple long/flat strategy:
    - Enter long when prob_up >= entry_th
    - Exit to flat when prob_up < exit_th
    PnL computed using next-step price change with fees/slippage.
    """
    df = df_probs.copy().reset_index(drop=True)
    df['pos'] = 0
    in_pos = False
    for i in range(len(df)):
        p = df.loc[i, prob_col]
        if not in_pos and p >= entry_th:
            in_pos = True
            df.loc[i, 'pos'] = 1
        elif in_pos and p < exit_th:
            in_pos = False
            df.loc[i, 'pos'] = 0
        else:
            df.loc[i, 'pos'] = 1 if in_pos else 0
    df['price_next'] = df[price_col].shift(-1)
    df['ret'] = (df['price_next'] - df[price_col]) / (df[price_col] + 1e-12)
    df['pos_change'] = df['pos'].diff().abs().fillna(0)
    trade_cost = (fee_bps + slip_bps) * 1e-4
    df['pnl'] = df['pos'] * df['ret'] - df['pos_change'] * trade_cost
    df['eq_curve'] = df['pnl'].cumsum()
    return df
