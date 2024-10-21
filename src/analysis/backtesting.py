"""
Core backtesting engine for algorithmic trading strategies.

This module provides the Backtester class which simulates trading
based on signal functions with realistic market conditions.
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine that executes trades based on signal functions.

    Supports long and short positions with commission and slippage.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0001):
        """
        Initialize the backtester.

        Args:
            data: OHLCV DataFrame with datetime index
            initial_capital: Starting capital
            commission: Commission per trade (as decimal)
            slippage: Slippage per trade (as decimal)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.shares = 0

        self.trades = []
        self.equity_curve = pd.DataFrame()

    def _calculate_position_size(self, capital: float, price: float) -> int:
        """Calculate number of shares to trade."""
        return int(capital / price)

    def _enter_position(self, idx: int, signal: int, price: float):
        """Enter a long or short position."""
        if signal == 1:
            cost = price * (1 + self.slippage + self.commission)
            self.shares = self._calculate_position_size(self.capital, cost)
            total_cost = self.shares * cost

            if self.shares > 0 and total_cost <= self.capital:
                self.capital -= total_cost
                self.position = 1
                self.entry_price = price

        elif signal == -1:
            cost = price * (1 - self.slippage + self.commission)
            self.shares = self._calculate_position_size(self.capital, cost)

            if self.shares > 0:
                self.capital += self.shares * price * (1 - self.slippage - self.commission)
                self.position = -1
                self.entry_price = price

    def _exit_position(self, idx: int, price: float):
        """Exit current position."""
        if self.position == 1:
            proceeds = self.shares * price * (1 - self.slippage - self.commission)
            self.capital += proceeds
            pnl = proceeds - (self.shares * self.entry_price)

        elif self.position == -1:
            cost = self.shares * price * (1 + self.slippage + self.commission)
            pnl = (self.shares * self.entry_price) - cost
            self.capital -= cost
        else:
            return

        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'shares': self.shares,
            'pnl': pnl,
            'position_type': 'long' if self.position == 1 else 'short'
        })

        self.position = 0
        self.shares = 0
        self.entry_price = 0

    def run_backtest(self, signal_function: Callable) -> pd.DataFrame:
        """
        Run backtest using provided signal function.

        Args:
            signal_function: Function that takes (data, idx) and returns signal

        Returns:
            DataFrame with trade history
        """
        logger.info("Starting backtest...")

        for idx in range(len(self.data)):
            signal = signal_function(self.data, idx)
            price = self.data['close'].iloc[idx]

            if self.position == 0 and signal != 0:
                self._enter_position(idx, signal, price)
            elif self.position != 0 and signal == -self.position:
                self._exit_position(idx, price)
                if signal != 0:
                    self._enter_position(idx, signal, price)

        if self.position != 0:
            self._exit_position(len(self.data) - 1, self.data['close'].iloc[-1])

        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return pd.DataFrame(self.trades)
