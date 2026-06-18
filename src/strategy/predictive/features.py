"""
Feature engineering for price prediction models.
Includes technical indicators and derived features for 1-minute kline data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class PriceFeatures:
    """Feature engineering for time series price prediction."""
    
    def __init__(self, lookback_window: int = 60):
        """
        Initialize feature generator.
        
        Args:
            lookback_window: Number of previous candles to use for features
        """
        self.lookback_window = lookback_window
    
    @staticmethod
    def add_returns(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """Add return features (log returns and simple returns)."""
        df['return'] = df[column].pct_change()
        df['log_return'] = np.log(df[column] / df[column].shift(1))
        return df
    
    @staticmethod
    def add_ohlc_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Add OHLC-based ratio features."""
        df['hl_ratio'] = df['high'] / df['low']
        df['co_ratio'] = df['close'] / df['open']
        df['cc_ratio'] = df['close'] / df['close'].shift(1)
        df['ho_ratio'] = df['high'] / df['open']
        df['lo_ratio'] = df['low'] / df['open']
        return df
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = None, column: str = 'close') -> pd.DataFrame:
        """Add Simple Moving Average indicators."""
        if periods is None:
            periods = [5, 10, 20, 50]
        
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = None, column: str = 'close') -> pd.DataFrame:
        """Add Exponential Moving Average indicators."""
        if periods is None:
            periods = [5, 10, 20, 50]
        
        for period in periods:
            df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_momentum(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add Momentum indicators."""
        if periods is None:
            periods = [5, 10, 20]
        
        for period in periods:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = 'close') -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0, column: str = 'close') -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * num_std)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * num_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add volume-based features."""
        if periods is None:
            periods = [5, 10, 20]
        
        # Volume moving averages
        for period in periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        df['obv'] = obv.cumsum()
        df['obv_sma'] = pd.Series(df['obv']).rolling(window=20).mean()
        
        return df
    
    @staticmethod
    def add_volatility(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add volatility features."""
        if periods is None:
            periods = [5, 10, 20, 60]
        
        for period in periods:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
        
        return df
    
    @staticmethod
    def add_correlation_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Add price-volume correlation and other correlation features."""
        df['price_volume_corr'] = df['close'].rolling(lookback).corr(df['volume'])
        
        return df
    
    def generate_all_features(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """Generate all available features."""
        df = df.copy()
        
        # Basic features
        df = self.add_returns(df)
        df = self.add_ohlc_ratios(df)
        
        # Moving averages
        df = self.add_sma(df, periods=[5, 10, 20, 50, 100])
        df = self.add_ema(df, periods=[5, 10, 20, 50, 100])
        
        # Momentum
        df = self.add_momentum(df, periods=[5, 10, 20, 60])
        
        # Technical indicators
        df = self.add_rsi(df, period=14)
        df = self.add_rsi(df, period=7)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        
        # Volume features
        df = self.add_volume_features(df)
        
        # Volatility
        df = self.add_volatility(df)
        
        # Correlation
        df = self.add_correlation_features(df, lookback=self.lookback_window)
        
        # Lagged price features (previous n candles)
        for lag in range(1, min(6, self.lookback_window + 1)):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['return'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Drop rows with NaN values if requested
        if dropna:
            df = df.dropna().reset_index(drop=True)
        
        return df
    
    @staticmethod
    def create_target(df: pd.DataFrame, horizon: int = 1, column: str = 'close') -> pd.DataFrame:
        """
        Create target variable (price direction/movement for next candle).
        
        Args:
            df: DataFrame with price data
            horizon: Number of candles into the future to predict
            column: Column to use for target (usually 'close')
        
        Returns:
            DataFrame with target columns added
        """
        df = df.copy()
        
        # Future close price
        df['future_close'] = df[column].shift(-horizon)
        
        # Price movement (binary classification: 1 if up, 0 if down)
        df['target_binary'] = (df['future_close'] > df[column]).astype(int)
        
        # Price change (regression)
        df['target_return'] = df['future_close'].pct_change(-1)  # Will be negative of the actual return
        df['target_return'] = (df[column].shift(-horizon) - df[column]) / df[column]  # Correct calculation
        
        # Price movement in absolute terms
        df['target_price_diff'] = df['future_close'] - df[column]
        
        return df


def prepare_data_for_modeling(
    df: pd.DataFrame,
    horizon: int = 1,
    lookback_window: int = 60,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Prepare data with all features and target variable.
    
    Args:
        df: Raw kline data
        horizon: Prediction horizon in candles
        lookback_window: Lookback window for feature generation
        dropna: Whether to drop NaN values
    
    Returns:
        Tuple of (DataFrame with features and target, feature names list, target column name)
    """
    # Generate features
    feature_gen = PriceFeatures(lookback_window=lookback_window)
    df_features = feature_gen.generate_all_features(df, dropna=False)
    
    # Create target
    df_features = feature_gen.create_target(df_features, horizon=horizon)
    
    # Drop NaN values
    if dropna:
        df_features = df_features.dropna().reset_index(drop=True)
    
    # Identify feature columns (exclude target and original OHLCV)
    exclude_cols = {
        'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
        'number_of_trades', 'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume', 'ignore',
        'symbol', 'open_time', 'close_time',
        'future_close', 'target_binary', 'target_return', 'target_price_diff'
    }
    
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    # Remove any remaining NaN columns
    feature_cols = [col for col in feature_cols if df_features[col].notna().any()]
    
    return df_features, feature_cols, 'target_binary'

