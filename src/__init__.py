"""
AlgoTrade - Algorithmic Trading Backtesting Framework

A robust framework for developing, backtesting, and analyzing trading strategies.

This package provides:
- Backtester: Core backtesting engine with signal-based trading
- Performance metrics: Sharpe, Sortino, drawdown, win rate, etc.
- Visualization tools: Equity curves, trade analysis, performance dashboards
- Technical indicators: Common indicators (SMA, EMA, RSI, MACD, etc.)
- Strategy framework: BaseStrategy abstract class for implementing strategies
"""

# Core backtesting engine
from .analysis.backtesting import Backtester

# Performance metrics
from .analysis.performance import calculate_cumulative_pnl, calculate_metrics

# Visualization tools
from .analysis.visualization import (
    plot_cum_pnl,
    plot_capital_curve,
    plot_trade_distribution,
    plot_equity_curve,
    plot_drawdown_curve,
    plot_trade_analysis,
    create_performance_dashboard
)

# Basic technical indicators
from .analysis.indicators import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
    stochastic,
    adx,
    volatility
)

# Strategy framework
from .strategies.base import BaseStrategy

__version__ = '1.0.0'

__all__ = [
    # Core engine
    'Backtester',

    # Strategy framework
    'BaseStrategy',

    # Performance metrics
    'calculate_cumulative_pnl',
    'calculate_metrics',

    # Visualization
    'plot_cum_pnl',
    'plot_capital_curve',
    'plot_trade_distribution',
    'plot_equity_curve',
    'plot_drawdown_curve',
    'plot_trade_analysis',
    'create_performance_dashboard',

    # Indicators
    'sma',
    'ema',
    'rsi',
    'macd',
    'bollinger_bands',
    'atr',
    'stochastic',
    'adx',
    'volatility'
]