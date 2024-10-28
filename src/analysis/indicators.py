"""
Basic technical indicators for trading strategies.

This module provides common technical indicators that can be used
with the backtesting framework. All indicators follow a consistent
interface accepting DataFrames and returning Series or DataFrames.
"""

import numpy as np
import pandas as pd


def sma(data: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA).

    Args:
        data (pd.DataFrame): DataFrame containing price data
        window (int): Window for SMA calculation
        column (str): Column name to use for calculation (default: 'close')

    Returns:
        pd.Series: SMA values
    """
    return data[column].rolling(window=window).mean()


def ema(data: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).

    Args:
        data (pd.DataFrame): DataFrame containing price data
        window (int): Window for EMA calculation
        column (str): Column name to use for calculation (default: 'close')

    Returns:
        pd.Series: EMA values
    """
    return data[column].ewm(span=window, adjust=False).mean()


def rsi(data: pd.DataFrame, window: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        data (pd.DataFrame): DataFrame containing price data
        window (int): The window for RSI calculation
        column (str): Column name to use for calculation (default: 'close')

    Returns:
        pd.Series: RSI values
    """
    delta = data[column].diff()

    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
        signal_period: int = 9, column: str = 'close') -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD).

    Args:
        data (pd.DataFrame): DataFrame containing price data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        column (str): Column name to use for calculation (default: 'close')

    Returns:
        pd.DataFrame: DataFrame with 'macd', 'signal', and 'histogram' columns
    """
    ema_fast = data[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data[column].ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    result = pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=data.index)

    return result


def bollinger_bands(data: pd.DataFrame, window: int = 20, std_dev: float = 2.0,
                  column: str = 'close') -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        data (pd.DataFrame): DataFrame containing price data
        window (int): Moving average window
        std_dev (float): Number of standard deviations for the bands
        column (str): Column name to use for calculation (default: 'close')

    Returns:
        pd.DataFrame: DataFrame with 'middle', 'upper', 'lower' and 'bandwidth' columns
    """
    middle_band = data[column].rolling(window=window).mean()
    std = data[column].rolling(window=window).std()

    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    bandwidth = (upper_band - lower_band) / middle_band

    result = pd.DataFrame({
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band,
        'bandwidth': bandwidth
    }, index=data.index)

    return result


def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR).

    Args:
        data (pd.DataFrame): DataFrame containing OHLC price data
        window (int): Window for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_values = tr.rolling(window=window).mean()

    return atr_values


def stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC price data
        k_period (int): K period
        d_period (int): D period

    Returns:
        pd.DataFrame: DataFrame with 'k' and 'd' columns
    """
    highest_high = data['high'].rolling(window=k_period).max()
    lowest_low = data['low'].rolling(window=k_period).min()
    k = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()

    result = pd.DataFrame({
        'k': k,
        'd': d
    }, index=data.index)

    return result


def adx(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX).

    Args:
        data (pd.DataFrame): DataFrame containing OHLC price data
        window (int): Window for ADX calculation

    Returns:
        pd.DataFrame: DataFrame with 'adx', 'di_plus', and 'di_minus' columns
    """
    high_diff = data['high'].diff()
    low_diff = data['low'].diff().multiply(-1)

    plus_dm = pd.Series(0, index=data.index)
    minus_dm = pd.Series(0, index=data.index)

    condition1 = (high_diff > low_diff) & (high_diff > 0)
    plus_dm[condition1] = high_diff[condition1]

    condition2 = (low_diff > high_diff) & (low_diff > 0)
    minus_dm[condition2] = low_diff[condition2]

    tr = atr(data, window)

    plus_di = 100 * (plus_dm.rolling(window=window).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / tr)

    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di

    dx = 100 * (di_diff / di_sum)
    adx_val = dx.rolling(window=window).mean()

    result = pd.DataFrame({
        'adx': adx_val,
        'di_plus': plus_di,
        'di_minus': minus_di
    }, index=data.index)

    return result


def volatility(data: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Calculate price volatility (standard deviation of returns).

    Args:
        data (pd.DataFrame): DataFrame containing price data
        window (int): Window for calculation
        column (str): Column name to use for calculation (default: 'close')

    Returns:
        pd.Series: Volatility values (annualized)
    """
    returns = data[column].pct_change()
    volatility_values = returns.rolling(window=window).std() * (252 ** 0.5)

    return volatility_values
