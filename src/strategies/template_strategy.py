"""
Template trading strategy to be used as a base for custom strategies.
This file should remain in the repository as a reference, but user-specific
strategies should be developed separately.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from src.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class TemplateStrategy(BaseStrategy):
    """
    Template strategy class that shows the structure of a strategy implementation.
    This serves as a guide for implementing your own strategies.
    """
    
    def __init__(self, data: pd.DataFrame, 
                fast_ma: int = 20, 
                slow_ma: int = 50,
                **kwargs):
        """
        Initialize the strategy with market data and parameters.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns and datetime index
            fast_ma (int): Fast moving average period
            slow_ma (int): Slow moving average period
        """
        super().__init__(data)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.parameters.update({
            'fast_ma': fast_ma,
            'slow_ma': slow_ma
        })
        
        # Add additional parameters from kwargs
        self.parameters.update(kwargs)
        
        logger.info(f"Initialized TemplateStrategy with parameters: {self.parameters}")
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for the strategy by calculating indicators.
        
        Returns:
            pd.DataFrame: Data with added indicators
        """
        # Make a copy to avoid modifying the original data
        data = self.data.copy()
        
        # Calculate moving averages
        data[f'fast_ma'] = data['close'].rolling(window=self.fast_ma).mean()
        data[f'slow_ma'] = data['close'].rolling(window=self.slow_ma).mean()
        
        # Calculate crossover signal
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1  # Buy signal
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1 # Sell signal
        
        # Calculate other indicators as needed
        # Example:
        # data['rsi'] = calculate_rsi(data['close'], window=14)
        
        return data
    
    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals for the data.
        
        Returns:
            pd.Series: Series of signals (1: buy, -1: sell, 0: hold)
        """
        # Prepare data with indicators
        prepared_data = self.prepare_data()
        
        # Drop NaN values that occur at the beginning due to rolling calculations
        prepared_data = prepared_data.dropna()
        
        # Extract signals
        signals = prepared_data['signal']
        
        # Store signals for later use in advanced backtesting
        self.signals = signals
        
        return signals
    
    def backtest(self) -> pd.DataFrame:
        """
        Run backtest of the strategy.
        
        Returns:
            pd.DataFrame: Tradebook containing all trades
        """
        # Prepare data with indicators
        prepared_data = self.prepare_data()
        
        # Initialize tracking variables
        tradenumber = 1
        position = 0  # 0: no position, 1: long, -1: short
        
        # Iterate through data
        for i in range(len(prepared_data)):
            current_row = prepared_data.iloc[i]
            
            # Skip NaN values that occur at the beginning due to rolling calculations
            if pd.isna(current_row['fast_ma']) or pd.isna(current_row['slow_ma']):
                continue
            
            current_time = current_row.name if isinstance(prepared_data.index, pd.DatetimeIndex) else current_row['datetime']
            close_price = current_row['close']
            signal = current_row['signal']
            
            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if position == -1:  # Close short position first
                    self._close_trade(tradenumber, current_time, close_price)
                
                # Open long position
                self._add_trade(tradenumber, current_time, close_price, "long")
                tradenumber += 1
                position = 1
                
            elif signal == -1 and position >= 0:  # Sell signal
                if position == 1:  # Close long position first
                    self._close_trade(tradenumber, current_time, close_price)
                
                # Open short position
                self._add_trade(tradenumber, current_time, close_price, "short")
                tradenumber += 1
                position = -1
        
        # Close any open position at the end of the backtest
        if len(self.tradebook) > 0 and self.tradebook['exit_time'].iloc[-1] == "":
            self._close_trade(tradenumber, prepared_data.index[-1], prepared_data['close'].iloc[-1])
            return self.tradebook
        
        return self.tradebook
    
    def optimize_parameters(self, initial_capital: float = 100000) -> Dict:
        """
        Run parameter optimization for the strategy.
        
        Args:
            initial_capital (float): Initial capital for the backtest
            
        Returns:
            Dict: Best parameters found
        """
        # Define parameter grid
        parameter_grid = {
            'fast_ma': [5, 10, 15, 20, 25],
            'slow_ma': [30, 40, 50, 60, 70],
        }
        
        # Define constraints
        constraints = {
            'sharpe_ratio': 0.5,  # Minimum acceptable Sharpe ratio
            'max_drawdown': -0.2  # Maximum acceptable drawdown
        }
        
        # Run optimization
        best_params, best_metrics = self.optimize(
            parameter_grid=parameter_grid,
            metric='sharpe_ratio',
            constraints=constraints
        )
        
        return best_params
    
    def run_strategy(self, initial_capital: float = 100000, 
                    optimize: bool = False,
                    visualize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the strategy with optional optimization and visualization.
        
        Args:
            initial_capital (float): Initial capital
            optimize (bool): Whether to run parameter optimization
            visualize (bool): Whether to visualize results
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tradebook and equity curve
        """
        if optimize:
            best_params = self.optimize_parameters(initial_capital)
            self.set_parameters(best_params)
            
            # Reinitialize with best parameters
            self.fast_ma = best_params['fast_ma']
            self.slow_ma = best_params['slow_ma']
        
        # Run advanced backtest
        tradebook, equity_curve = self.run_advanced_backtest(
            initial_capital=initial_capital
        )
        
        # Visualize results if requested
        if visualize:
            self.visualize_results(tradebook, equity_curve, dashboard=True)
        
        return tradebook, equity_curve 