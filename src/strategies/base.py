from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the strategy with market data.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
        """
        self.data = data
        self.tradebook = pd.DataFrame(columns=['entry_time', 'entry_price', 
                                             'exit_time', 'exit_price', 'position'])
        self.parameters = {}
        self.signals = None
    
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals for the data.
        
        Returns:
            pd.Series: Series of signals (1: buy, -1: sell, 0: hold)
        """
        pass
    
    @abstractmethod
    def backtest(self) -> pd.DataFrame:
        """
        Run backtest of the strategy.
        
        Returns:
            pd.DataFrame: Tradebook containing all trades
        """
        pass
    
    def set_parameters(self, parameters: Dict) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters (Dict): Dictionary of parameters
        """
        self.parameters = parameters
        logger.info(f"Set parameters: {parameters}")
    
    def optimize(self, parameter_grid: Dict[str, List], 
                metric: str = 'sharpe_ratio',
                constraints: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            parameter_grid (Dict[str, List]): Dictionary of parameter names and values to try
            metric (str): Metric to optimize for
            constraints (Dict, optional): Constraints on metrics
                
        Returns:
            Tuple[Dict, Dict]: Best parameters and corresponding metrics
        """
        logger.info(f"Starting parameter optimization...")
        
        from itertools import product
        import time
        
        # Generate all parameter combinations
        keys = parameter_grid.keys()
        values = parameter_grid.values()
        combinations = list(product(*values))
        
        best_params = None
        best_metrics = None
        best_metric_value = float('-inf')
        
        # Iterate over all parameter combinations
        for i, combination in enumerate(combinations):
            start_time = time.time()
            current_params = dict(zip(keys, combination))
            
            # Set parameters and run backtest
            self.set_parameters(current_params)
            try:
                signals = self.generate_signals()
                tradebook = self.backtest()
                
                # Calculate metrics
                from src.analysis.performance import calculate_metrics
                metrics = calculate_metrics(tradebook)
                
                # Check constraints
                if constraints is not None:
                    constraint_violated = False
                    for constraint_metric, constraint_value in constraints.items():
                        if metrics[constraint_metric] < constraint_value:
                            constraint_violated = True
                            break
                    
                    if constraint_violated:
                        continue
                
                # Check if this is the best so far
                if metrics[metric] > best_metric_value:
                    best_metric_value = metrics[metric]
                    best_params = current_params.copy()
                    best_metrics = metrics.copy()
                
                logger.info(f"Combination {i+1}/{len(combinations)}, {current_params}, "
                           f"{metric}: {metrics[metric]:.4f}, time: {time.time() - start_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error in optimization for {current_params}: {str(e)}")
        
        logger.info(f"Optimization complete. Best parameters: {best_params}")
        logger.info(f"Best metrics: {best_metrics}")
        
        return best_params, best_metrics
    
    def _add_trade(self, tradenumber: int, current_time: str, 
                   entry_price: float, position_type: str) -> None:
        """
        Add a new trade to the tradebook.
        
        Args:
            tradenumber (int): Trade number
            current_time (str): Time of trade
            entry_price (float): Entry price
            position_type (str): 'long' or 'short'
        """
        self.tradebook.loc[tradenumber] = [current_time, entry_price, "", 0.00, position_type]
    
    def _close_trade(self, tradenumber: int, current_time: str, 
                    exit_price: float) -> None:
        """
        Close an existing trade.
        
        Args:
            tradenumber (int): Trade number
            current_time (str): Time of trade
            exit_price (float): Exit price
        """
        self.tradebook.loc[tradenumber - 1, 'exit_time'] = current_time
        self.tradebook.loc[tradenumber - 1, 'exit_price'] = exit_price
    
    def run_advanced_backtest(self, initial_capital: float = 100000, 
                             commission: float = 0.00075, 
                             slippage: float = 0.00005,
                             risk_per_trade: float = 0.02,
                             stop_loss: Optional[float] = None,
                             take_profit: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run an advanced backtest using the backtesting module.
        
        Args:
            initial_capital (float): Initial capital
            commission (float): Commission per trade
            slippage (float): Slippage per trade
            risk_per_trade (float): Risk per trade as percentage of capital
            stop_loss (float, optional): Stop loss percentage
            take_profit (float, optional): Take profit percentage
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tradebook and equity curve
        """
        from src.analysis.backtesting import Backtester
        
        # Generate signals if not already generated
        if self.signals is None:
            self.signals = self.generate_signals()
        
        # Define signal function for backtester
        def signal_function(data, idx):
            return self.signals.iloc[idx]
        
        # Run backtest
        backtester = Backtester(
            data=self.data,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            risk_per_trade=risk_per_trade,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        tradebook = backtester.run_backtest(signal_function=signal_function)
        
        return tradebook, backtester.equity_curve
    
    def visualize_results(self, tradebook: pd.DataFrame, 
                        equity_curve: pd.DataFrame, 
                        benchmark: Optional[pd.DataFrame] = None,
                        dashboard: bool = False) -> None:
        """
        Visualize backtest results.
        
        Args:
            tradebook (pd.DataFrame): Tradebook with trades
            equity_curve (pd.DataFrame): Equity curve
            benchmark (pd.DataFrame, optional): Benchmark data
            dashboard (bool): Whether to show comprehensive dashboard
        """
        from src.analysis.visualization import (
            plot_equity_curve, plot_drawdown_curve, plot_trade_analysis,
            plot_monthly_returns_heatmap, plot_rolling_statistics,
            plot_price_with_trades, create_performance_dashboard
        )
        
        if dashboard:
            create_performance_dashboard(
                equity_curve=equity_curve, 
                tradebook=tradebook,
                price_data=self.data,
                benchmark=benchmark
            )
        else:
            # Individual plots
            plot_equity_curve(equity_curve, benchmark)
            plot_drawdown_curve(equity_curve)
            plot_trade_analysis(tradebook)
            plot_monthly_returns_heatmap(equity_curve)
            plot_rolling_statistics(equity_curve)
            plot_price_with_trades(self.data, tradebook) 