#!/usr/bin/env python
"""
Example script demonstrating how to use the AlgoTrade backtesting framework.

This example shows two methods:
1. Direct Backtester usage with a signal function (recommended for simple strategies)
2. BaseStrategy class usage (recommended for complex strategies with optimization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import framework components
from src import Backtester, BaseStrategy, sma, ema, create_performance_dashboard

def load_sample_data():
    """
    Load or generate sample data for testing.
    
    Returns:
        pd.DataFrame: Sample price data
    """
    try:
        # Try to load data from a CSV file if available
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            logger.info(f"Loaded sample data from {data_path}")
            return df
    except Exception as e:
        logger.warning(f"Could not load sample data: {str(e)}")
    
    # Generate synthetic data
    logger.info("Generating synthetic data for testing")
    np.random.seed(42)
    
    # Generate dates
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price series with trends, seasonality, and noise
    n = len(dates)
    
    # Base price
    base_price = 100
    
    # Trend component (random walk with drift)
    trend = np.cumsum(np.random.normal(0.0005, 0.01, n))
    
    # Seasonality (weekly and annual)
    weekly = 0.005 * np.sin(np.linspace(0, 2*n*np.pi/7, n))
    annual = 0.05 * np.sin(np.linspace(0, 2*np.pi, n))
    
    # Random events (occasional jumps)
    jumps = np.zeros(n)
    jump_points = np.random.choice(range(n), size=5, replace=False)
    jumps[jump_points] = np.random.normal(0, 0.05, 5)
    jumps = np.cumsum(jumps)
    
    # Combine components
    price = base_price * np.exp(trend + weekly + annual + jumps)
    
    # Create OHLC data
    close = price
    high = close * np.exp(np.random.normal(0.001, 0.005, n))
    low = close * np.exp(np.random.normal(-0.001, 0.005, n))
    open_price = close * np.exp(np.random.normal(0, 0.003, n))
    volume = np.random.normal(1000000, 200000, n)
    volume = np.abs(volume) + 500000
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Save for future use
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv'))
    
    return df

def method1_direct_backtester():
    """
    Method 1: Direct Backtester usage with signal function.

    This is the recommended approach for simple strategies where you
    just need to define signal generation logic.
    """
    logger.info("\n" + "="*60)
    logger.info("METHOD 1: Direct Backtester Usage")
    logger.info("="*60)

    # Load sample data
    data = load_sample_data()
    logger.info(f"Loaded data with {len(data)} bars")

    # Precompute indicators for signal function
    fast_ma = sma(data, window=10)
    slow_ma = sma(data, window=30)

    # Define signal function
    def moving_average_crossover_signal(data, idx):
        """
        Simple moving average crossover strategy.

        Returns:
            1: Buy signal (fast MA > slow MA)
           -1: Sell signal (fast MA < slow MA)
            0: Hold
        """
        if idx >= len(fast_ma):
            return 0

        if fast_ma.iloc[idx] > slow_ma.iloc[idx]:
            return 1  # Bullish
        elif fast_ma.iloc[idx] < slow_ma.iloc[idx]:
            return -1  # Bearish
        return 0

    # Initialize backtester
    backtester = Backtester(
        data=data,
        initial_capital=100000,
        commission=0.00075,       # 0.075% commission per trade
        slippage=0.00005,         # 0.005% slippage
        risk_per_trade=0.02,      # Risk 2% of capital per trade
        stop_loss=0.05,           # 5% stop loss
        take_profit=0.10          # 10% take profit
    )

    # Run backtest
    logger.info("Running backtest...")
    tradebook = backtester.run_backtest(signal_function=moving_average_crossover_signal)

    logger.info(f"Completed {len(tradebook)} trades")
    logger.info(f"Final equity: ${backtester.equity_curve['equity'].iloc[-1]:,.2f}")

    # Visualize results
    create_performance_dashboard(
        equity_curve=backtester.equity_curve,
        tradebook=tradebook,
        price_data=data
    )

    return tradebook, backtester.equity_curve


def method2_basestrategy_class():
    """
    Method 2: Using BaseStrategy class for complex strategies.

    This approach is better for strategies that need:
    - Parameter optimization
    - Complex signal generation logic
    - Multiple indicators
    - Custom backtesting logic
    """
    logger.info("\n" + "="*60)
    logger.info("METHOD 2: BaseStrategy Class Usage")
    logger.info("="*60)

    # Load sample data
    data = load_sample_data()
    logger.info(f"Loaded data with {len(data)} bars")

    # Define a custom strategy class
    class SimpleMovingAverageStrategy(BaseStrategy):
        """
        A simple moving average crossover strategy using BaseStrategy.
        """

        def __init__(self, data, fast_period=10, slow_period=30):
            super().__init__(data)
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.parameters.update({
                'fast_period': fast_period,
                'slow_period': slow_period
            })

        def generate_signals(self):
            """Generate trading signals based on MA crossover."""
            # Calculate indicators
            fast_ma = sma(self.data, self.fast_period)
            slow_ma = sma(self.data, self.slow_period)

            # Generate signals
            signals = pd.Series(0, index=self.data.index)
            signals[fast_ma > slow_ma] = 1   # Bullish
            signals[fast_ma < slow_ma] = -1  # Bearish

            self.signals = signals
            return signals

        def backtest(self):
            """Run backtest using the framework."""
            tradebook, equity_curve = self.run_advanced_backtest(
                initial_capital=100000,
                commission=0.00075,
                slippage=0.00005,
                risk_per_trade=0.02,
                stop_loss=0.05,
                take_profit=0.10
            )
            return tradebook, equity_curve

    # Initialize strategy
    strategy = SimpleMovingAverageStrategy(data, fast_period=10, slow_period=30)

    # Run backtest
    logger.info("Running backtest...")
    tradebook, equity_curve = strategy.backtest()

    logger.info(f"Completed {len(tradebook)} trades")
    logger.info(f"Final equity: ${equity_curve['equity'].iloc[-1]:,.2f}")

    # Visualize results
    strategy.visualize_results(tradebook, equity_curve, dashboard=True)

    return tradebook, equity_curve


def run_all_examples():
    """Run all example methods."""
    logger.info("AlgoTrade Backtesting Framework - Examples")
    logger.info("="*60)

    # Run Method 1
    tradebook1, equity1 = method1_direct_backtester()

    # Run Method 2
    tradebook2, equity2 = method2_basestrategy_class()

    logger.info("\n" + "="*60)
    logger.info("Examples completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    logger.info("Starting AlgoTrade framework examples")
    run_all_examples()
    logger.info("All examples completed") 