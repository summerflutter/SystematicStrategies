from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


def compute_beta_spread(y: pd.Series, x: pd.Series):
    df = pd.concat([y,x], axis=1).dropna()
    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]

    X = add_constant(x_clean)
    model = OLS(y_clean, X).fit()
    beta = model.params.iloc[-1]
    spread = y_clean - beta*x_clean
    return beta, spread


def engle_granger_coint_test(y:pd.Series, x:pd.Series) -> dict:
    score, pvalue, _ = coint(y, x)
    beta, spread = compute_beta_spread(y, x)

    adf_res = adfuller(spread, autolag = "AIC")
    adf_pvalue = adf_res[1]

    return {
        "pvalue": pvalue,
        "score": score,
        "hedge_ratio": beta,
        "spread": spread,
        "adf_resid_pvalue": adf_pvalue,
    }


def estimate_half_life(spread:pd.Series) -> Optional[float]:
    s = spread.dropna()
    if len(s)<30:
        return None

    s_lag = s.shift(1).dropna()
    s_curr = s.loc[s_lag.index]

    X = add_constant(s_lag.values)
    model = OLS(s_curr.values, X).fit()
    b = model.params[1]

    if b <= 0 or b>=1:
        return None

    hl = -np.log(2) / np.log(b)
    return float(hl)


@dataclass
class StatArbSignal:
    """
    - Class representing the statistical arbitrage signal
    - It'll be applied on the spread
    - Going forward need to add the margin rate in the strategy
    """

    def __init__(self, beta, spread_long_entry, spread_short_entry, spread_long_exit_tp, spread_long_exit_sl, spread_short_exit_tp, spread_short_exit_sl, cooldown_bars = 20):
        self.beta = beta
        self.spread_long_entry = spread_long_entry
        self.spread_short_entry = spread_short_entry
        self.spread_long_exit_tp = spread_long_exit_tp
        self.spread_long_exit_sl = spread_long_exit_sl
        self.spread_short_exit_tp = spread_short_exit_tp
        self.spread_short_exit_sl = spread_short_exit_sl
        self.cooldown_bars = cooldown_bars

        return

    def __repr__(self):
        return (f'hedge_ratio (beta) = {self.beta}, \n'
                f'spread_long_entry={self.spread_long_entry},\n'
                f'spread_short_entry={self.spread_short_entry},\n'
                f'spread_long_exit_tp={self.spread_long_exit_tp},\n'
                f'spread_long_exit_sl={self.spread_long_exit_sl},\n'
                f'spread_short_exit_tp={self.spread_short_exit_tp}, \n'
                f'spread_short_exit_sl={self.spread_short_exit_sl}, \n'
                f'cooldown_bars={self.cooldown_bars}')


def generate_spread_signal(y:pd.Series,
                           x:pd.Series,
                            entry_z: float = 2.0,
                            exit_z_tp: float = 0.5,
                           exit_z_sl: float = 3.0,
                           cooldown_bars: int = 20) -> StatArbSignal:
    """
    The simpliest pairs trading strategy based on the z-score of spread
    - z > entry_z: short spread (short y, long x)
    - z < entry_z: long spread (long y, short x)
    - |z| < exit_z: flat

    Return:
        - signal: statistical arbitrage signal class
        - position: +1/-1/0 which is the direct on y (position of x is contrary)
    """
    beta, spread = compute_beta_spread(y, x)
    s = spread.dropna()
    s_std = s.std()
    s_mean = s.mean()
    z = (s - s_mean) / s_std

    # get the spread level to build positions
    spread_long_entry = (-entry_z)*s_std + s_mean
    spread_long_exit_tp = (-exit_z_tp)*s_std + s_mean
    spread_long_exit_sl = (-exit_z_sl)*s_std + s_mean
    spread_short_entry = entry_z*s_std + s_mean
    spread_short_exit_tp = exit_z_tp*s_std + s_mean
    spread_short_exit_sl = (exit_z_sl)*s_std + s_mean

    signal = StatArbSignal(beta, spread_long_entry, spread_short_entry, spread_long_exit_tp,spread_long_exit_sl, spread_short_exit_tp, spread_short_exit_sl, cooldown_bars = cooldown_bars)

    return signal


def build_spread_signal_from_stats(
    beta: float,
    spread_mean: float,
    spread_std: float,
    entry_z: float = 2.0,
    exit_z_tp: float = 0.5,
    exit_z_sl: float = 3.0,
    cooldown_bars: int = 20,
) -> StatArbSignal:
    """Build StatArbSignal from pre-computed spread statistics."""
    if spread_std <= 0:
        spread_std = 1.0
    return StatArbSignal(
        beta,
        (-entry_z) * spread_std + spread_mean,
        entry_z * spread_std + spread_mean,
        (-exit_z_tp) * spread_std + spread_mean,
        (-exit_z_sl) * spread_std + spread_mean,
        exit_z_tp * spread_std + spread_mean,
        exit_z_sl * spread_std + spread_mean,
        cooldown_bars=cooldown_bars,
    )



def backtest_spread_signal(
        y:pd.Series,
        x:pd.Series,
        signal:StatArbSignal,
        init_nav: float = 1000.0,
        leverage_long: float = 1.0,
        leverage_short: float = 1.0,
        max_margin_utilization: float = 1.0,
        trading_fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        funding_rate_per_period_X: float = 0.0,
        funding_rate_per_period_Y: float = 0.0,
        funding_series_X: pd.Series | None = None,
        funding_series_Y: pd.Series | None = None) -> pd.DataFrame:
    """
    This function applies spread signal to spread and computes the portfolio value
    Note: y and x should be clean, no null data
    :param y: price series for asset y
    :param x: price series for asset x
    :param signal: the strategy signal generated, including the entry and exit level for long/short
    :param init_nav:the initial net asset value (cash) to invest. Assuming half capital to long, and half to short
    :return: dataframe including: positions, nav
    """

    df = pd.concat([y, x], axis=1).dropna()

    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]

    spread = y_clean - signal.beta * x_clean
    idx = df.index
    n = len(idx)

    df["beta"] = signal.beta
    beta = signal.beta
    df["spread"] = spread

    # compute the triggering levels here
    long_entry_arr = [(i < signal.spread_long_entry) and (i > signal.spread_long_exit_sl) for i in df['spread']]
    short_entry_arr = [(i > signal.spread_short_entry) and (i < signal.spread_short_exit_sl) for i in df['spread']]
    positive_position_clear_arr = [(i >= signal.spread_long_exit_tp) or (i <= signal.spread_long_exit_sl) for i in df['spread']]
    negative_position_clear_arr = [(i <= signal.spread_short_exit_tp) or (i >= signal.spread_short_exit_sl) for i in df['spread']]
    cooldown_triggered_arr = [(i >= signal.spread_short_exit_sl) or (i <= signal.spread_long_exit_sl) for i in df['spread']]

    # ===== Position variables =====
    units_Y_arr = np.zeros(n, dtype=float)
    units_X_arr = np.zeros(n, dtype=float)
    units_Y_prev_arr = np.zeros(n, dtype=float)
    units_X_prev_arr = np.zeros(n, dtype=float)
    dY_arr = np.zeros(n, dtype=float)
    dX_arr = np.zeros(n, dtype=float)
    prev_Y_arr = np.zeros(n, dtype=float)
    prev_X_arr = np.zeros(n, dtype=float)

    # ===== Account variables =====
    nav_arr = np.zeros(n, dtype=float)
    pnl_price_arr = np.zeros(n, dtype=float)
    pnl_arr = np.zeros(n, dtype=float)
    cum_pnl_arr = np.zeros(n, dtype=float)
    fee_arr = np.zeros(n, dtype=float)
    funding_arr = np.zeros(n, dtype=float)

    # ===== Notional / margin / leverage =====
    notional_long_arr = np.zeros(n, dtype=float)
    notional_short_arr = np.zeros(n, dtype=float)
    gross_exposure_arr = np.zeros(n, dtype=float)
    margin_long_arr = np.zeros(n, dtype=float)
    margin_short_arr = np.zeros(n, dtype=float)
    margin_total_arr = np.zeros(n, dtype=float)

    # ===== State variables =====
    position_state_arr = np.zeros(n, dtype=np.int8)
    cooldown_count_arr = np.zeros(n, dtype=np.int32)
    leverage_arr = np.zeros(n, dtype=float)

    # Initial state
    nav = init_nav
    cum_pnl = 0.0
    pnl_price = 0.0
    units_Y = 0.0
    units_X = 0.0
    position_state = 0
    cooldown_count = 0

    leg_cost_bps = trading_fee_bps + slippage_bps

    # =============== Start the loop ======================
    prev_y = y.iloc[0]
    prev_x = x.iloc[0]

    for i, t in enumerate(idx):
        Y_t = y_clean.iloc[i]
        X_t = x_clean.iloc[i]
        spread_t = spread.iloc[i]

        rate_X = float(funding_series_X.iloc[i]) if funding_series_X is not None else funding_rate_per_period_X
        rate_Y = float(funding_series_Y.iloc[i]) if funding_series_Y is not None else funding_rate_per_period_Y

        # 1) firstly update pnl and nav according to the position and price move from last bar
        if i>0:
            dY = Y_t - prev_y
            dX = X_t - prev_x

            # pnl from price move
            pnl_price = units_Y * dY + units_X * dX

            prev_Y_arr[i] = prev_y
            prev_X_arr[i] = prev_x
            units_Y_prev_arr[i] = prev_y
            units_X_prev_arr[i] = prev_x
            dY_arr[i] = dY
            dX_arr[i] = dX

            # Funding: positive rate => longs pay, shorts receive
            notional_Y = units_Y * Y_t
            notional_X = units_X * X_t
            funding_t = rate_Y * notional_Y + rate_X * notional_X

            # transaction fee: 0 unless opening/closing below
            fee_t = 0.0

            pnl_t = pnl_price - funding_t
            cum_pnl += pnl_t
            nav += pnl_t
        else:
            pnl_t = 0.0
            funding_t = 0.0
            fee_t = 0.0


        # 2) decide exit/entry depending if there is position or not
        # Note: all the transactions are done with current price Y_t and X_t

        # check if we're currently in cooldown state
        if cooldown_triggered_arr[i]:
            cooldown_count = signal.cooldown_bars

        # --- if there is position: check if we need to exit ---
        if position_state != 0:
            if position_state == 1:
                # check if spread_long_exit is triggered
                exit_flag = positive_position_clear_arr[i]
            else:
                exit_flag = negative_position_clear_arr[i]

            if exit_flag:
                # exit the position: clear all the units and calculate the fee
                notional_long = max(0, units_X) * X_t + max(0, units_Y) * Y_t
                notional_short = abs(min(0, units_X) * X_t) + abs(min(0, units_Y) * Y_t)
                trade_notional = notional_long + notional_short
                fee_close = trade_notional * leg_cost_bps / 10000.0

                nav -= fee_close
                fee_t += fee_close
                cum_pnl -= fee_close

                units_X = 0
                units_Y = 0
                position_state = 0


        # --- No position: check if open position ---
        elif (position_state == 0) and (cooldown_count==0):
            long_entry_flag = long_entry_arr[i]
            short_entry_flag = short_entry_arr[i]

            # compute the margin requirement for 1 unit of spread
            if long_entry_flag or short_entry_flag:
                if long_entry_flag:
                    # long spread: +1Y, -beta X
                    margin_per_unit = Y_t / leverage_long + beta * X_t / leverage_short
                elif short_entry_flag:
                    # short spread: -1Y, +beta X
                    margin_per_unit = Y_t / leverage_short + beta * X_t / leverage_long

                # get the maximum margin in the account
                max_margin = nav * max_margin_utilization

                if margin_per_unit > 0 and max_margin > 0:
                    units_spread = max_margin / margin_per_unit # the largest unit of spread position
                else:
                    units_spread = 0.0


                if units_spread > 0:
                    # set the position depending on long/short
                    if long_entry_flag:
                        position_state = 1
                        units_Y += units_spread
                        units_X -= beta * units_spread
                    elif short_entry_flag:
                        position_state = -1
                        units_Y -= units_spread
                        units_X += beta * units_spread

                    # compute the fee to open position
                    trade_notional_open = abs(units_X) * X_t + abs(units_Y) * Y_t
                    fee_open = trade_notional_open * leg_cost_bps / 10000.0

                    nav -= fee_open
                    fee_t += fee_open
                    cum_pnl -= fee_open

        elif (position_state == 0) and (cooldown_count>0):
            cooldown_count -= 1

        # 3) record all the states to df
        # the notional and margin
        notional_long_t = abs(max(0, units_X) * X_t) + abs(max(0, units_Y) * Y_t)
        notional_short_t = abs(min(0, units_X) * X_t) + abs(min(0, units_Y) * Y_t)
        gross_exposure_t = notional_long_t + notional_short_t

        margin_long_t = notional_long_t / leverage_long
        margin_short_t = notional_short_t / leverage_short
        margin_total_t = margin_long_t + margin_short_t

        units_Y_arr[i] = units_Y
        units_X_arr[i] = units_X
        position_state_arr[i] = position_state
        cooldown_count_arr[i] = cooldown_count

        pnl_arr[i] = pnl_t
        pnl_price_arr[i] = pnl_price
        cum_pnl_arr[i] = cum_pnl
        nav_arr[i] = nav
        fee_arr[i] = fee_t
        funding_arr[i] = funding_t

        notional_long_arr[i] = notional_long_t
        notional_short_arr[i] = notional_short_t
        gross_exposure_arr[i] = gross_exposure_t
        margin_long_arr[i] = margin_long_t
        margin_short_arr[i] = margin_short_t
        margin_total_arr[i] = margin_total_t

        if nav > 0:
            leverage_arr[i] = gross_exposure_t / nav
        else:
            leverage_arr[i] = np.nan

        # update the price for prev
        prev_y = Y_t
        prev_x = X_t

    # add the columns to df
    df["units_Y"] = units_Y_arr
    df["units_X"] = units_X_arr
    df["units_Y_prev"] = units_Y_prev_arr
    df["units_X_prev"] = units_X_prev_arr
    df["dY"] = dY_arr
    df["dX"] = dX_arr
    df["prev_Y"] = prev_Y_arr
    df["prev_X"] = prev_X_arr

    df["nav"] = nav_arr
    df["pnl_price"] = pnl_price_arr
    df["pnl"] = pnl_arr
    df["cum_pnl"] = cum_pnl_arr
    df["fee"] = fee_arr
    df["funding"] = funding_arr

    df["notional_long"] = notional_long_arr
    df["notional_short"] = notional_short_arr
    df["gross_exposure"] = gross_exposure_arr
    df["margin_long"] = margin_long_arr
    df["margin_short"] = margin_short_arr
    df["margin_total"] = margin_total_arr

    df["position_state"] = position_state_arr
    df["cooldown_count"] = cooldown_count_arr
    df["leverage"] = leverage_arr

    return df
