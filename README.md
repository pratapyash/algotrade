# AlgoTrade: Backtesting Engine Framework

A robust, production-ready backtesting engine for algorithmic trading strategies.

## Key Features

- Signal-based backtesting with flexible strategy interface
- Realistic trading simulation (commission, slippage, position sizing)
- Risk management (stop-loss, take-profit, risk-per-trade)
- Comprehensive performance metrics (Sharpe, Sortino, drawdown, etc.)
- Rich visualization suite with performance dashboards
- Basic technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Strategy optimization framework with grid search

## Project Structure

```
algotrade/
├── src/
│   ├── analysis/
│   │   ├── backtesting.py      # Core Backtester class
│   │   ├── performance.py      # Performance metrics
│   │   ├── visualization.py    # Plotting tools
│   │   └── indicators.py       # Technical indicators
│   ├── strategies/
│   │   ├── base.py             # BaseStrategy ABC
│   │   └── template_strategy.py # Example template
│   └── __init__.py             # Public API
│
├── examples/
│   └── backtest_example.py     # Usage examples
├── tests/                      # Unit tests
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/algotrade.git
cd algotrade
```

2. Create and activate environment:

```bash
conda create -n trade python=3.8
conda activate trade
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Method 1: Direct Backtester Usage (Recommended)

The core approach is to use the `Backtester` class with a signal function:

```python
import pandas as pd
from src import Backtester, sma

# Load your data
data = pd.read_csv('data/price_data.csv')

# Define your signal function
def my_signal_function(data, idx):
    """
    Generate trading signals based on price data.

    Returns:
        1: Buy signal (go long)
       -1: Sell signal (go short)
        0: Hold/no action
    """
    fast_ma = sma(data, window=20)
    slow_ma = sma(data, window=50)

    if fast_ma.iloc[idx] > slow_ma.iloc[idx]:
        return 1  # Bullish
    elif fast_ma.iloc[idx] < slow_ma.iloc[idx]:
        return -1  # Bearish
    return 0

# Run backtest
backtester = Backtester(
    data=data,
    initial_capital=100000,
    commission=0.00075,      # 0.075% per trade
    slippage=0.00005,        # 0.005% slippage
    risk_per_trade=0.02,     # Risk 2% per trade
    stop_loss=0.05,          # 5% stop loss
    take_profit=0.10         # 10% take profit
)

tradebook = backtester.run_backtest(signal_function=my_signal_function)

# Visualize results
from src import create_performance_dashboard
create_performance_dashboard(
    equity_curve=backtester.equity_curve,
    tradebook=tradebook,
    price_data=data
)
```

### Method 2: Using BaseStrategy Class

For more complex strategies, inherit from `BaseStrategy`:

```python
from src import BaseStrategy
from src.analysis.indicators import ema, rsi

class MyStrategy(BaseStrategy):
    def __init__(self, data, fast_period=10, slow_period=30):
        super().__init__(data)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self):
        """Generate trading signals."""
        fast_ema = ema(self.data, self.fast_period)
        slow_ema = ema(self.data, self.slow_period)
        rsi_values = rsi(self.data, 14)

        signals = pd.Series(0, index=self.data.index)
        signals[(fast_ema > slow_ema) & (rsi_values < 70)] = 1
        signals[(fast_ema < slow_ema) & (rsi_values > 30)] = -1

        return signals

    def backtest(self):
        """Run backtest using the framework."""
        return self.run_advanced_backtest(
            initial_capital=100000,
            commission=0.00075,
            slippage=0.00005
        )

# Use the strategy
strategy = MyStrategy(data, fast_period=10, slow_period=30)
tradebook, equity_curve = strategy.backtest()
```

## Available Indicators

```python
from src import (
    sma,              # Simple Moving Average
    ema,              # Exponential Moving Average
    rsi,              # Relative Strength Index
    macd,             # MACD
    bollinger_bands,  # Bollinger Bands
    atr,              # Average True Range
    stochastic,       # Stochastic Oscillator
    adx,              # Average Directional Index
    volatility        # Price Volatility
)
```

## Performance Metrics

The framework calculates comprehensive performance metrics:

```python
from src import calculate_metrics

metrics = calculate_metrics(tradebook)

# Available metrics:
# - Sharpe Ratio
# - Sortino Ratio
# - Maximum Drawdown
# - Win Rate
# - Profit Factor
# - Average Win/Loss
# - Total Trades
# - Total Return
```

## Running the Example

```bash
python examples/backtest_example.py
```

This demonstrates both usage methods with synthetic data.

## Parameter Optimization

Built-in grid search optimization:

```python
strategy = MyStrategy(data)

parameter_grid = {
    'fast_period': [5, 10, 15, 20],
    'slow_period': [30, 40, 50, 60]
}

best_params, best_metrics = strategy.optimize(
    parameter_grid=parameter_grid,
    metric='sharpe_ratio',
    constraints={'max_drawdown': -0.20}  # Max 20% drawdown
)
```

## Testing

Run tests to verify the framework:

```bash
python -m pytest tests/
```

## Architecture

The framework follows a clean separation of concerns:

- **Backtester**: Core engine that executes trades based on signals
- **BaseStrategy**: Abstract interface for strategy implementations
- **Indicators**: Technical analysis functions
- **Performance**: Metrics calculation and analysis
- **Visualization**: Charts and dashboards
