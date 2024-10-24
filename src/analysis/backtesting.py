"""
Backtesting module for trading strategies.
This module provides a robust backtesting framework that can be used by any strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
from datetime import datetime

# Assuming performance.py is in the same directory
from . import performance  # Added for performance calculations

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    """
    A robust backtesting engine for trading strategies.
    
    This class handles the backtesting of any trading strategy by simulating
    the execution of trades based on strategy signals.
    """
    
    def __init__(self, 
                data: pd.DataFrame, 
                initial_capital: float = 100000, 
                commission: float = 0.00075, 
                slippage: float = 0.00005,
                risk_per_trade: float = 0.02,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                compound: bool = True,
                enter_on: str = 'close',
                exit_on: str = 'close'):
        """
        Initialize the backtester.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns and datetime index
            initial_capital (float): Initial capital for backtest
            commission (float): Commission per trade as a percentage
            slippage (float): Slippage per trade as a percentage
            risk_per_trade (float): Risk per trade as a percentage of capital
            stop_loss (float, optional): Stop loss percentage
            take_profit (float, optional): Take profit percentage
            compound (bool): Whether to compound returns or reset to initial capital after each trade
            enter_on (str): Whether to enter on 'open' or 'close'
            exit_on (str): Whether to exit on 'open' or 'close'
        """
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.compound = compound
        self.enter_on = enter_on
        self.exit_on = exit_on
        
        # Input validation
        if enter_on not in ['open', 'close']:
            raise ValueError("enter_on must be either 'open' or 'close'")
        if exit_on not in ['open', 'close']:
            raise ValueError("exit_on must be either 'open' or 'close'")
        
        # Initialize performance tracking
        self.tradebook = pd.DataFrame(columns=[
            'entry_time', 'entry_price', 'exit_time', 'exit_price', 
            'position', 'size', 'pnl', 'pnl_pct', 'cum_pnl', 'capital'
        ])
        
        self.equity_curve_list = []  # Renamed from equity_curve to avoid confusion with DataFrame
        self.processed_equity_df = pd.DataFrame()  # Will store final equity df with drawdown/runup

        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.entry_time = None
        self.trade_number = 1
        
        # peak_capital and max_drawdown_at_trade_closure are for the specific metric
        # of drawdown observed at trade closure points.
        self.peak_capital_for_trade_dd = initial_capital 
        self.max_drawdown_at_trade_closure = 0 

        # ZeltaGold style metrics
        self.gross_profit = 0
        self.gross_loss = 0
        self.win_trades = 0
        self.losing_trades = 0
        
        logger.info(f"Initialized Backtester with {len(data)} data points")
        
    def _calculate_position_size(self, price: float, stop_price: Optional[float] = None) -> float:
        """
        Calculate position size based on risk management parameters.
        
        Args:
            price (float): Current price for entry
            stop_price (float, optional): Stop loss price if specified
            
        Returns:
            float: Size of position in units
        """
        # MODIFIED: Use self.initial_capital if not self.compound for risk calculation
        capital_for_risk = self.initial_capital if not self.compound else self.current_capital
        risk_amount = capital_for_risk * self.risk_per_trade
        
        if stop_price is not None:
            # Ensure price and stop_price are not equal to avoid division by zero
            if price == stop_price:
                 logger.warning(f"Price ({price}) and stop_price ({stop_price}) are equal. Cannot calculate position size based on stop. Defaulting to risk_amount / price.")
                 return risk_amount / price if price > 0 else 0

            risk_per_share = abs(price - stop_price)  # Amount risked per share
            if risk_per_share > 0:
                return risk_amount / risk_per_share
            else:  # Should be caught by price == stop_price, but as a fallback
                return risk_amount / price if price > 0 else 0
        
        # Default sizing if no stop is specified (invest risk_amount directly)
        return risk_amount / price if price > 0 else 0
    
    def _update_trade_drawdown_metrics(self) -> None:  # Renamed to clarify its scope
        """
        Update maximum drawdown observed at trade closure points.
        This is based on current capital relative to its peak at trade times.
        """
        if self.current_capital > self.peak_capital_for_trade_dd:
            self.peak_capital_for_trade_dd = self.current_capital
        
        current_trade_drawdown = (self.peak_capital_for_trade_dd - self.current_capital) / self.peak_capital_for_trade_dd \
                                 if self.peak_capital_for_trade_dd > 0 else 0
        
        if current_trade_drawdown > self.max_drawdown_at_trade_closure:
            self.max_drawdown_at_trade_closure = current_trade_drawdown
            
    def _execute_entry(self, time_val: Union[str, datetime], price: float, 
                      position_type: str, size: Optional[float] = None) -> None:  # time renamed to time_val
        """
        Execute an entry order.
        
        Args:
            time_val (str or datetime): Entry time
            price (float): Entry price
            position_type (str): 'long' or 'short'
            size (float, optional): Position size in units, if None will be calculated
        """
        # Apply slippage to entry
        adjusted_price = price * (1 + self.slippage) if position_type == 'long' else price * (1 - self.slippage)
        
        # Calculate position size if not provided
        if size is None:
            # Determine stop_price for position sizing if self.stop_loss is set
            stop_level = None
            if self.stop_loss:
                stop_level = adjusted_price * (1 - self.stop_loss) if position_type == 'long' else adjusted_price * (1 + self.stop_loss)
            size = self._calculate_position_size(adjusted_price, stop_level)

        if size <= 0:
            logger.warning(f"Attempted to enter {position_type} with size {size:.4f}. Skipping trade.")
            return
        
        # Record entry
        self.entry_price = adjusted_price
        self.entry_time = time_val
        self.current_position = 1 if position_type == 'long' else -1
        self.position_size = size
        
        # Apply commission
        self.current_capital -= adjusted_price * size * self.commission
        
        logger.info(f"Entered {position_type.upper()} position at {time_val}, price: {adjusted_price:.2f}, size: {size:.4f}")
    
    def _execute_exit(self, time_val: Union[str, datetime], price: float, position_type: str) -> None:  # time renamed to time_val
        """
        Execute an exit order.
        
        Args:
            time_val (str or datetime): Exit time
            price (float): Exit price
            position_type (str): 'long' or 'short'
        """
        # Apply slippage to exit
        adjusted_price = price * (1 - self.slippage) if position_type == 'long' else price * (1 + self.slippage)
        
        # Calculate PnL
        if position_type == 'long':
            pnl = (adjusted_price - self.entry_price) * self.position_size
            pnl_pct = (adjusted_price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0
        else:  # short
            pnl = (self.entry_price - adjusted_price) * self.position_size
            pnl_pct = (self.entry_price - adjusted_price) / self.entry_price if self.entry_price != 0 else 0
        
        # Apply commission on exit
        pnl -= (adjusted_price * self.position_size * self.commission)  # Commission on exit value

        # Update capital
        self.current_capital += (self.position_size * self.entry_price) + pnl
        
        # Update win/loss statistics
        if pnl > 0:
            self.win_trades += 1
            self.gross_profit += pnl
        else:
            self.losing_trades += 1
            self.gross_loss += pnl
        
        # Record trade in tradebook
        cum_pnl = self.current_capital - self.initial_capital
        self._update_trade_drawdown_metrics()  # Updates self.max_drawdown_at_trade_closure
        
        self.tradebook.loc[self.trade_number] = [
            self.entry_time, self.entry_price, time_val, adjusted_price, 
            position_type, self.position_size, pnl, pnl_pct, cum_pnl, self.current_capital
        ]
        
        # Update trade counter and reset position
        self.trade_number += 1
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        
        logger.info(f"Exited {position_type.upper()} position at {time_val}, price: {adjusted_price:.2f}, PnL: {pnl:.2f} ({pnl_pct:.2%})")
    
    def run_backtest(self, 
                     signal_function: Callable[[pd.DataFrame, int], int],
                     tearsheet: bool = True) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        Run the backtest using the provided signal function.
        
        Args:
            signal_function: Function that returns 1 (buy), -1 (sell), or 0 (hold)
                            given the dataframe and current index
            tearsheet (bool): Whether to print a performance tearsheet after backtest
            
        Returns:
            Tuple containing:
            - pd.DataFrame: Tradebook with all executed trades
            - Dict: Performance metrics
            - pd.DataFrame: Equity curve dataframe (self.processed_equity_df)
        """
        start_time_val = time.time()  # time renamed to time_val
        logger.info("Starting backtest...")
        
        # Initialize equity_curve_list with initial capital for the first data point's timestamp
        self.equity_curve_list = [self.initial_capital] * (1 if len(self.data) > 0 else 0)

        for i in range(1, len(self.data)):  # Loop from the second data point
            # Get current bar data
            current_time = self.data.index[i]
            current_price_for_sl_tp = self.data['close'].iloc[i] 
            signal_decision_price = self.data[self.exit_on].iloc[i] if self.exit_on in self.data.columns else self.data['close'].iloc[i]
            entry_trigger_price = self.data[self.enter_on].iloc[i] if self.enter_on in self.data.columns else self.data['close'].iloc[i]

            # Add current capital to equity curve for the PREVIOUS bar's state
            if self.current_position != 0:
                stop_hit = False
                if self.stop_loss is not None:
                    if (self.current_position == 1 and 
                        current_price_for_sl_tp <= self.entry_price * (1 - self.stop_loss)):
                        self._execute_exit(current_time, current_price_for_sl_tp, 'long')
                        stop_hit = True
                    elif (self.current_position == -1 and 
                          current_price_for_sl_tp >= self.entry_price * (1 + self.stop_loss)):
                        self._execute_exit(current_time, current_price_for_sl_tp, 'short')
                        stop_hit = True
                
                if not stop_hit and self.take_profit is not None:
                    if (self.current_position == 1 and 
                        current_price_for_sl_tp >= self.entry_price * (1 + self.take_profit)):
                        self._execute_exit(current_time, current_price_for_sl_tp, 'long')
                        stop_hit = True
                    elif (self.current_position == -1 and 
                          current_price_for_sl_tp <= self.entry_price * (1 - self.take_profit)):
                        self._execute_exit(current_time, current_price_for_sl_tp, 'short')
                        stop_hit = True
                
                if stop_hit:
                    self.equity_curve_list.append(self.current_capital)
                    continue
            
            signal = signal_function(self.data, i) 
            
            if signal == 1 and self.current_position <= 0:  # Buy signal
                if self.current_position == -1:  # Close short position first
                    self._execute_exit(current_time, signal_decision_price, 'short')
                
                self._execute_entry(current_time, entry_trigger_price, 'long')
                
            elif signal == -1 and self.current_position >= 0:  # Sell signal
                if self.current_position == 1:  # Close long position first
                    self._execute_exit(current_time, signal_decision_price, 'long')
                
                self._execute_entry(current_time, entry_trigger_price, 'short')
            
            self.equity_curve_list.append(self.current_capital)
        
        if self.current_position == 1:
            self._execute_exit(self.data.index[-1], self.data['close'].iloc[-1], 'long')
        elif self.current_position == -1:
            self._execute_exit(self.data.index[-1], self.data['close'].iloc[-1], 'short')
        
        if len(self.equity_curve_list) == len(self.data.index) and len(self.data.index) > 0:
            equity_series = pd.Series(self.equity_curve_list, index=self.data.index, name='equity')
        elif len(self.data.index) == 0:
            equity_series = pd.Series(dtype=float, name='equity')
        else:
            logger.warning(f"Equity curve list length ({len(self.equity_curve_list)}) "
                           f"does not match data index length ({len(self.data.index)}). "
                           f"Attempting to align. This may indicate an issue in equity recording.")
            if len(self.equity_curve_list) > len(self.data.index):
                equity_series = pd.Series(self.equity_curve_list[:len(self.data.index)], index=self.data.index, name='equity')
            elif len(self.equity_curve_list) < len(self.data.index) and len(self.equity_curve_list) > 0:
                padding_needed = len(self.data.index) - len(self.equity_curve_list)
                padded_equity = self.equity_curve_list + [self.equity_curve_list[-1]] * padding_needed
                equity_series = pd.Series(padded_equity, index=self.data.index, name='equity')
            else:
                equity_series = pd.Series([self.initial_capital]*len(self.data.index), index=self.data.index, name='equity')

        self.processed_equity_df = pd.DataFrame(equity_series)
        if not self.processed_equity_df.empty:
            self.processed_equity_df['drawdown'] = performance.calculate_drawdown_series(self.processed_equity_df['equity'])
            self.processed_equity_df['runup'] = performance.calculate_runup_series(self.processed_equity_df['equity'])
        else:
            self.processed_equity_df = pd.DataFrame(columns=['equity', 'drawdown', 'runup'])

        logger.info(f"Backtest completed in {time.time() - start_time_val:.2f} seconds")
        
        performance_metrics = self.generate_performance_report()
        
        if tearsheet:
            pass 

        return self.tradebook, performance_metrics, self.processed_equity_df
    
    def generate_performance_report(self) -> Dict:
        """
        Generate and print performance metrics.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if len(self.tradebook) == 0 and self.processed_equity_df.empty:
            logger.warning("No trades executed and no equity curve data for backtest report.")
            return {}
        
        total_trades = len(self.tradebook)
        winning_trades = self.win_trades
        losing_trades = self.losing_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win_pct = self.tradebook[self.tradebook['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss_pct = self.tradebook[self.tradebook['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(self.gross_profit / self.gross_loss) if self.gross_loss != 0 else float('inf')
        
        if not self.processed_equity_df.empty and 'equity' in self.processed_equity_df.columns and len(self.processed_equity_df['equity']) > 0:
            equity_values = self.processed_equity_df['equity']
            total_return = (equity_values.iloc[-1] - equity_values.iloc[0]) / equity_values.iloc[0] if equity_values.iloc[0] != 0 else 0
            
            if len(equity_values.index) > 1:
                days = (equity_values.index[-1] - equity_values.index[0]).days
                years = days / 365.25
                annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
            else:
                annualized_return = total_return

            daily_returns = equity_values.pct_change().dropna()
            sharpe_ratio = performance.calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 1 else 0
            sortino_ratio = performance.calculate_sortino_ratio(daily_returns) if len(daily_returns) > 1 else 0
            
            max_drawdown_equity_pct = abs(self.processed_equity_df['drawdown'].min()) if not self.processed_equity_df['drawdown'].empty else 0
            
            buy_and_hold_return = ((self.data['close'].iloc[-1] - self.data['close'].iloc[0]) / self.data['close'].iloc[0]) \
                                  if len(self.data['close']) > 1 and self.data['close'].iloc[0] != 0 else 0
        else:
            total_return = 0.0; annualized_return = 0.0; sharpe_ratio = 0.0; sortino_ratio = 0.0; max_drawdown_equity_pct = 0.0; buy_and_hold_return = 0.0

        avg_holding_duration = pd.Timedelta(0)
        if total_trades > 0 and 'entry_time' in self.tradebook.columns and 'exit_time' in self.tradebook.columns:
            valid_trades_for_duration = self.tradebook.dropna(subset=['entry_time', 'exit_time'])
            if not valid_trades_for_duration.empty:
                entry_times = pd.to_datetime(valid_trades_for_duration['entry_time'])
                exit_times = pd.to_datetime(valid_trades_for_duration['exit_time'])
                durations = exit_times - entry_times
                if len(durations) > 0:
                    avg_holding_duration = durations.mean()
        
        logger.info("\n----- PERFORMANCE REPORT -----")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annualized_return:.2%}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Gross Profit: {self.gross_profit:.2f}")
        logger.info(f"Gross Loss: {self.gross_loss:.2f}")
        logger.info(f"Average Win %: {avg_win_pct:.2%}") 
        logger.info(f"Average Loss %: {avg_loss_pct:.2%}") 
        logger.info(f"Max Drawdown (Equity based): {max_drawdown_equity_pct:.2f}%") 
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {sortino_ratio:.2f}")
        logger.info(f"Average Holding Duration: {avg_holding_duration}")
        logger.info(f"Buy & Hold Return: {buy_and_hold_return:.2%}")
        logger.info(f"Max Drawdown (at trade closure): {self.max_drawdown_at_trade_closure*100:.2f}%")
        logger.info("-----------------------------\n")
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'max_drawdown_equity_pct': max_drawdown_equity_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_holding_duration': str(avg_holding_duration),
            'buy_and_hold_return': buy_and_hold_return,
            'largest_win_pnl': self.tradebook['pnl'].max() if total_trades > 0 else 0,
            'largest_loss_pnl': self.tradebook['pnl'].min() if total_trades > 0 else 0,
            'max_drawdown_at_trade_closure': self.max_drawdown_at_trade_closure
        }
        
    def plot_equity_curve(self, benchmark: Optional[pd.DataFrame] = None, save: bool = False, plot: bool = True):
        """
        Plot equity curve of the strategy with optional benchmark comparison.
        Relies on self.processed_equity_df.
        
        Args:
            benchmark (pd.DataFrame, optional): DataFrame with datetime index to use as benchmark
            save (bool): Whether to save the plot to file
            plot (bool): Whether to display the plot
        """
        if self.processed_equity_df.empty or 'equity' not in self.processed_equity_df.columns:
            logger.warning("No equity curve data to plot (processed_equity_df is empty or missing 'equity' column)")
            return
            
        plt.figure(figsize=(14, 7))
        
        equity_to_plot = self.processed_equity_df['equity']
        plt.plot(equity_to_plot.index, equity_to_plot, label='Strategy')
        
        if benchmark is not None and not benchmark.empty:
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                try:
                    benchmark.index = pd.to_datetime(benchmark.index)
                except Exception as e:
                    logger.error(f"Could not convert benchmark index to DatetimeIndex: {e}")
                    benchmark = None
            
            if benchmark is not None:
                aligned_benchmark_series = None
                if equity_to_plot.index.equals(benchmark.index):
                    numeric_cols_benchmark = benchmark.select_dtypes(include=np.number)
                    if not numeric_cols_benchmark.empty:
                        aligned_benchmark_series = numeric_cols_benchmark.iloc[:, 0]
                else:
                    reindexed_benchmark = benchmark.reindex(equity_to_plot.index).ffill().bfill()
                    numeric_cols_reindexed = reindexed_benchmark.select_dtypes(include=np.number)
                    if not numeric_cols_reindexed.empty:
                        aligned_benchmark_series = numeric_cols_reindexed.iloc[:, 0]

                if aligned_benchmark_series is not None and not aligned_benchmark_series.empty and aligned_benchmark_series.iloc[0] != 0:
                    norm_factor = equity_to_plot.iloc[0] / aligned_benchmark_series.iloc[0]
                    plt.plot(aligned_benchmark_series.index, aligned_benchmark_series * norm_factor, label='Benchmark', alpha=0.7)
                elif aligned_benchmark_series is not None and (aligned_benchmark_series.empty or aligned_benchmark_series.iloc[0] == 0):
                    logger.warning("Benchmark data is empty or starts at zero after alignment, cannot plot normalized benchmark.")
                elif aligned_benchmark_series is None:
                    logger.warning("No numeric column found in benchmark data for plotting.")

        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig('./output/equity_curve.png')
        if plot:
            plt.show()
            
    def plot_drawdown(self, save: bool = False, plot: bool = True):
        """
        Plot drawdown curve. Relies on self.processed_equity_df.
        
        Args:
            save (bool): Whether to save the plot to file
            plot (bool): Whether to display the plot
        """
        if self.processed_equity_df.empty or 'drawdown' not in self.processed_equity_df.columns:
            logger.warning("No drawdown data to plot (processed_equity_df is empty or missing 'drawdown' column)")
            return
            
        drawdown_to_plot = self.processed_equity_df['drawdown']
            
        plt.figure(figsize=(14, 7))
        plt.fill_between(drawdown_to_plot.index, drawdown_to_plot, 0, color='r', alpha=0.3)
        plt.plot(drawdown_to_plot.index, drawdown_to_plot, color='r', alpha=0.8)
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('./output/drawdown.png')
        if plot:
            plt.show()
            
    def plot_trade_analysis(self, save: bool = False, plot: bool = True):
        """
        Plot trade analysis charts (win/loss, avg profit/loss, etc).
        Relies on self.tradebook.
        
        Args:
            save (bool): Whether to save the plot to file
            plot (bool): Whether to display the plot
        """
        if len(self.tradebook) == 0:
            logger.warning("No trades to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Win/Loss Breakdown
        win_mask = self.tradebook['pnl'] > 0
        win_count = win_mask.sum()
        loss_count = (~win_mask).sum()
        
        axes[0, 0].bar(['Wins', 'Losses'], [win_count, loss_count], color=['g', 'r'])
        axes[0, 0].set_title('Win/Loss Breakdown')
        axes[0, 0].grid(True, alpha=0.3)
        if win_count > 0: axes[0, 0].text(0, win_count/2, f"{win_count}", ha='center', va='center')
        if loss_count > 0: axes[0, 0].text(1, loss_count/2, f"{loss_count}", ha='center', va='center')
        
        # 2. Win Rate Over Time
        if len(self.tradebook) > 0:
            trade_numbers = range(1, len(self.tradebook) + 1)
            cumulative_wins = np.cumsum(win_mask.values)
            win_rate_series = cumulative_wins / trade_numbers
            
            axes[0, 1].plot(trade_numbers, win_rate_series, 'b-')
            axes[0, 1].set_title('Win Rate Over Time')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        else:
             axes[0, 1].set_title('Win Rate Over Time (No Trades)')

        # 3. Average Profit vs Loss (using pnl_pct)
        avg_win_pct = self.tradebook.loc[win_mask, 'pnl_pct'].mean() if win_count > 0 else 0
        avg_loss_pct = self.tradebook.loc[~win_mask, 'pnl_pct'].mean() if loss_count > 0 else 0
        
        axes[1, 0].bar(['Avg Win %', 'Avg Loss %'], [avg_win_pct*100, avg_loss_pct*100], color=['g', 'r'])
        axes[1, 0].set_title('Average Profit vs Loss (%)')
        axes[1, 0].set_ylabel('Return %')
        axes[1, 0].grid(True, alpha=0.3)
        if avg_win_pct != 0 : axes[1, 0].text(0, avg_win_pct*50, f"{avg_win_pct:.2%}", ha='center', va='center')
        if avg_loss_pct != 0 : axes[1, 0].text(1, avg_loss_pct*50, f"{avg_loss_pct:.2%}", ha='center', va='center')
        
        # 4. Cumulative PnL (from tradebook)
        if not self.tradebook.empty:
            axes[1, 1].plot(self.tradebook.index, self.tradebook['cum_pnl'], 'k-')
            axes[1, 1].set_title('Cumulative PnL (Trade-based)')
            axes[1, 1].set_xlabel('Trade Number (Tradebook Index)')
            axes[1, 1].set_ylabel('Cumulative PnL')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1,1].set_title('Cumulative PnL (No Trades)')

        plt.tight_layout()
        
        if save:
            plt.savefig('./output/trade_analysis.png')
        if plot:
            plt.show()
            
    def plot_entry_exit(self, window: Optional[int] = None, save: bool = False, plot: bool = True):
        """
        Plot price chart with entry and exit points.
        Relies on self.tradebook and self.data.
        
        Args:
            window (int, optional): Number of bars to show (most recent)
            save (bool): Whether to save the plot to file
            plot (bool): Whether to display the plot
        """
        if len(self.tradebook) == 0:
            logger.warning("No trades to plot for entry/exit analysis")
            return
            
        plot_data = self.data.copy()
        if window is not None and window < len(plot_data):
            plot_data = plot_data.iloc[-window:]
            
        plt.figure(figsize=(14, 7))
        
        plt.plot(plot_data.index, plot_data['close'], color='black', linewidth=1, label='Close Price')
        
        trades_in_window = self.tradebook[
            (pd.to_datetime(self.tradebook['entry_time']) >= plot_data.index.min()) |
            (pd.to_datetime(self.tradebook['exit_time']) <= plot_data.index.max())
        ]

        long_entries_plot = []
        long_exits_plot = []
        short_entries_plot = []
        short_exits_plot = []
        
        for _, trade in trades_in_window.iterrows():
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            if entry_time >= plot_data.index.min() and entry_time <= plot_data.index.max():
                if trade['position'] == 'long':
                    long_entries_plot.append((entry_time, trade['entry_price']))
                else:
                    short_entries_plot.append((entry_time, trade['entry_price']))
                    
            if exit_time >= plot_data.index.min() and exit_time <= plot_data.index.max():
                if trade['position'] == 'long':
                    long_exits_plot.append((exit_time, trade['exit_price']))
                else:
                    short_exits_plot.append((exit_time, trade['exit_price']))
        
        if long_entries_plot:
            times, prices = zip(*long_entries_plot)
            plt.scatter(times, prices, marker='^', color='green', s=100, label='Long Entry', zorder=5)
            
        if long_exits_plot:
            times, prices = zip(*long_exits_plot)
            plt.scatter(times, prices, marker='v', color='red', s=100, label='Long Exit', zorder=5)
            
        if short_entries_plot:
            times, prices = zip(*short_entries_plot)
            plt.scatter(times, prices, marker='v', color='maroon', s=100, label='Short Entry', zorder=5)
            
        if short_exits_plot:
            times, prices = zip(*short_exits_plot)
            plt.scatter(times, prices, marker='^', color='lime', s=100, label='Short Exit', zorder=5)
        
        plt.title('Price Chart with Entry and Exit Points')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig('./output/entry_exit.png')
        if plot:
            plt.show()
                
    def create_performance_dashboard(self, benchmark: Optional[pd.DataFrame] = None, save: bool = False, plot: bool = True):
        """
        Create a comprehensive performance dashboard.
        This now relies on self.processed_equity_df for equity-based plots.
        """
        self.plot_equity_curve(benchmark, save, plot)
        self.plot_drawdown(save, plot)
        self.plot_trade_analysis(save, plot)
        self.plot_entry_exit(save=save, plot=plot)