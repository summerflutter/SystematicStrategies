import pandas as pd

from src.data.data_download import (
    download_spot_klines_monthly,
    download_perp_klines_monthly,
)

DEFAULT_ASSETS = ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","SUI"]
DEFAULT_ASSETS = ["BTC"]
DEFAULT_SYMBOLS = [a + "USDT" for a in DEFAULT_ASSETS]

def get_last_month_end() -> pd.Timestamp:
    """
    Return the end of last month in UTC.
    Example: 2026-04-19 -> 2026-03-31 00:00:00+00:00
    """
    now = pd.Timestamp.utcnow()
    last_month_end = (now.replace(day=1) - pd.Timedelta(days=1)).normalize()
    return last_month_end


def main():
    """
    Download and save spot and perp data for selected underlyings from 2024
    """

    # get the last month end
    last_month_end = get_last_month_end()

    # Download spot
    download_spot_klines_monthly(
        symbols = DEFAULT_SYMBOLS,
        interval = "1m",
        start = pd.Timestamp("2020-01-01"),
        end = get_last_month_end(),
    )


if __name__ == "__main__":
    main()

