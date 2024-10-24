import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Optional, Tuple, Union, Dict
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import mplfinance as mpf
from datetime import datetime, timedelta

def plot_cum_pnl(tradebook: pd.DataFrame) -> None:
    """
    Plot cumulative PnL over time.
    
    Args:
        tradebook (pd.DataFrame): Tradebook with PnL calculations
    """
    plt.figure(figsize=(12, 6))
    plt.plot(tradebook['cum_pnl'], label='Cumulative PnL')
    plt.title('Cumulative PnL Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative PnL')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_capital_curve(tradebook: pd.DataFrame) -> None:
    """
    Plot capital curve over time.
    
    Args:
        tradebook (pd.DataFrame): Tradebook with PnL calculations
    """
    plt.figure(figsize=(12, 6))
    plt.plot(tradebook['capital'], label='Capital')
    plt.title('Capital Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Capital')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trade_distribution(tradebook: pd.DataFrame) -> None:
    """
    Plot distribution of trade PnLs.
    
    Args:
        tradebook (pd.DataFrame): Tradebook with PnL calculations
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(tradebook['pnl'], bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Trade PnL Distribution')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_equity_curve(equity_curve: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> None:
    """
    Plot equity curve of the strategy with optional benchmark comparison.
    
    Args:
        equity_curve (pd.DataFrame): DataFrame with datetime index and 'equity' column
        benchmark (pd.DataFrame, optional): DataFrame with datetime index and column for benchmark values
    """
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve.index, equity_curve['equity'], label='Strategy')
    
    if benchmark is not None:
        # Normalize benchmark to same starting value as strategy
        norm_factor = equity_curve['equity'].iloc[0] / benchmark.iloc[0]
        plt.plot(benchmark.index, benchmark * norm_factor, label='Benchmark', alpha=0.7)
    
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    # Format y-axis to show currency
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()

def plot_drawdown_curve(equity_curve: pd.DataFrame) -> None:
    """
    Plot drawdown curve.
    
    Args:
        equity_curve (pd.DataFrame): DataFrame with datetime index and 'equity' column
    """
    # If drawdown is already provided in the DataFrame, use it
    if 'drawdown' in equity_curve.columns:
        drawdown = equity_curve['drawdown']
    else:
        # Calculate drawdown
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max * 100
    
    plt.figure(figsize=(14, 7))
    plt.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='r', alpha=0.8)
    plt.title('Drawdown Curve')
    plt.xlabel('Date')
    plt.ylabel('Drawdown %')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_runup_curve(equity_curve: pd.DataFrame) -> None:
    """
    Plot runup curve.
    
    Args:
        equity_curve (pd.DataFrame): DataFrame with datetime index and runup column
    """
    # If runup is already provided in the DataFrame, use it
    if 'runup' in equity_curve.columns:
        runup = equity_curve['runup']
    else:
        # Calculate runup
        rolling_min = equity_curve['equity'].cummin()
        runup = (equity_curve['equity'] - rolling_min) / rolling_min * 100
    
    plt.figure(figsize=(14, 7))
    plt.fill_between(runup.index, runup, 0, color='g', alpha=0.3)
    plt.plot(runup.index, runup, color='g', alpha=0.8)
    plt.title('Runup Curve')
    plt.xlabel('Date')
    plt.ylabel('Runup %')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_monthly_returns_heatmap(equity_curve: pd.DataFrame) -> None:
    """
    Plot monthly returns as a heatmap.
    
    Args:
        equity_curve (pd.DataFrame): DataFrame with datetime index and 'equity' column
    """
    # Calculate daily returns
    daily_returns = equity_curve['equity'].pct_change().dropna()
    
    # Convert to monthly returns
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table for the heatmap
    monthly_pivot = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot_table = monthly_pivot.pivot('Year', 'Month', 'Return')
    
    # Draw the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', center=0, 
               linewidths=1, cbar_kws={'label': 'Monthly Return'})
    
    # Configure labels and title
    plt.title('Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.gca().set_xticklabels(month_names)
    
    plt.tight_layout()
    plt.show()

def plot_trade_analysis(tradebook: pd.DataFrame) -> None:
    """
    Plot trade analysis charts (win/loss, avg profit/loss, duration).
    
    Args:
        tradebook (pd.DataFrame): Tradebook with PnL calculations
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Win/Loss Breakdown
    win_mask = tradebook['pnl'] > 0
    win_count = win_mask.sum()
    loss_count = (~win_mask).sum()
    
    axes[0, 0].bar(['Wins', 'Losses'], [win_count, loss_count], color=['g', 'r'])
    axes[0, 0].set_title('Win/Loss Breakdown')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0, win_count/2, f"{win_count}", ha='center')
    axes[0, 0].text(1, loss_count/2, f"{loss_count}", ha='center')
    
    # 2. Win Rate Over Time
    trade_numbers = range(1, len(tradebook) + 1)
    cumulative_wins = np.cumsum(win_mask)
    win_rate_series = cumulative_wins / trade_numbers
    
    axes[0, 1].plot(trade_numbers, win_rate_series, 'b-')
    axes[0, 1].set_title('Win Rate Over Time')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # 3. Average Profit vs Loss
    avg_win = tradebook.loc[win_mask, 'pnl_pct'].mean() if win_count > 0 else 0
    avg_loss = tradebook.loc[~win_mask, 'pnl_pct'].mean() if loss_count > 0 else 0
    
    axes[1, 0].bar(['Avg Win', 'Avg Loss'], [avg_win, avg_loss], color=['g', 'r'])
    axes[1, 0].set_title('Average Profit vs Loss')
    axes[1, 0].set_ylabel('Return %')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0, avg_win/2, f"{avg_win:.2%}", ha='center')
    axes[1, 0].text(1, avg_loss/2, f"{avg_loss:.2%}", ha='center')
    
    # 4. Trade Duration
    if 'entry_time' in tradebook.columns and 'exit_time' in tradebook.columns:
        durations = pd.to_datetime(tradebook['exit_time']) - pd.to_datetime(tradebook['entry_time'])
        durations = durations.dt.total_seconds() / (60 * 60 * 24)  # Convert to days
        
        axes[1, 1].hist(durations, bins=20, alpha=0.7)
        axes[1, 1].set_title('Trade Duration Distribution')
        axes[1, 1].set_xlabel('Duration (days)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no duration data, plot cumulative PnL instead
        axes[1, 1].plot(tradebook.index, tradebook['cum_pnl'], 'k-')
        axes[1, 1].set_title('Cumulative PnL')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative PnL')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_rolling_statistics(equity_curve: pd.DataFrame, window: int = 20) -> None:
    """
    Plot rolling performance statistics.
    
    Args:
        equity_curve (pd.DataFrame): DataFrame with datetime index and 'equity' column
        window (int): Rolling window size
    """
    # Calculate daily returns
    returns = equity_curve['equity'].pct_change().dropna()
    
    # Calculate rolling statistics
    rolling_return = returns.rolling(window=window).mean() * 252  # Annualized
    rolling_vol = returns.rolling(window=window).std() * (252 ** 0.5)  # Annualized
    rolling_sharpe = rolling_return / rolling_vol
    
    # Calculate rolling max drawdown
    rolling_dd = pd.Series(index=returns.index)
    for i in range(len(returns) - window + 1):
        equity_slice = equity_curve['equity'].iloc[i:i+window]
        peak = equity_slice.cummax()
        drawdown = (equity_slice - peak) / peak
        rolling_dd.iloc[i+window-1] = drawdown.min() * 100
    
    # Create the plots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Plot rolling returns
    axes[0].plot(rolling_return.index, rolling_return, 'b-')
    axes[0].set_title(f'Rolling {window}-Day Annualized Return')
    axes[0].set_ylabel('Return (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot rolling volatility
    axes[1].plot(rolling_vol.index, rolling_vol, 'r-')
    axes[1].set_title(f'Rolling {window}-Day Annualized Volatility')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot rolling Sharpe ratio
    axes[2].plot(rolling_sharpe.index, rolling_sharpe, 'g-')
    axes[2].set_title(f'Rolling {window}-Day Sharpe Ratio')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].grid(True, alpha=0.3)
    
    # Plot rolling max drawdown
    axes[3].plot(rolling_dd.index, rolling_dd, 'k-')
    axes[3].set_title(f'Rolling {window}-Day Max Drawdown')
    axes[3].set_ylabel('Drawdown (%)')
    axes[3].grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_candlestick_with_trades(price_data: pd.DataFrame, 
                                tradebook: pd.DataFrame,
                                title: str = 'Price Chart with Trades',
                                lookback_days: Optional[int] = None) -> None:
    """
    Plot candlestick chart with trade entry and exit points.
    
    Args:
        price_data (pd.DataFrame): DataFrame with OHLCV data
        tradebook (pd.DataFrame): Tradebook with entry/exit details
        title (str): Chart title
        lookback_days (int, optional): Number of days to look back from the last date
    """
    # Prepare data for mplfinance
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data = price_data.copy()
        price_data.index = pd.to_datetime(price_data.index)
    
    # Filter by lookback period if specified
    if lookback_days is not None:
        end_date = price_data.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        price_data = price_data.loc[start_date:end_date]
    
    # Prepare trade markers
    markers = []
    
    # Process long entries
    long_entries = tradebook[tradebook['position'] == 'long']
    for _, trade in long_entries.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        if entry_time in price_data.index:
            markers.append(mpf.make_addplot([trade['entry_price']], type='scatter', 
                                           markersize=100, marker='^', color='green', 
                                           ax=None, date=entry_time))
        
        if exit_time in price_data.index:
            markers.append(mpf.make_addplot([trade['exit_price']], type='scatter', 
                                           markersize=100, marker='v', color='red', 
                                           ax=None, date=exit_time))
    
    # Process short entries
    short_entries = tradebook[tradebook['position'] == 'short']
    for _, trade in short_entries.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        if entry_time in price_data.index:
            markers.append(mpf.make_addplot([trade['entry_price']], type='scatter', 
                                           markersize=100, marker='v', color='red', 
                                           ax=None, date=entry_time))
        
        if exit_time in price_data.index:
            markers.append(mpf.make_addplot([trade['exit_price']], type='scatter', 
                                           markersize=100, marker='^', color='green', 
                                           ax=None, date=exit_time))
    
    # Plot candlestick chart with markers
    mpf.plot(price_data, type='candle', style='yahoo', 
             title=title, 
             figsize=(14, 7), 
             addplot=markers if markers else None,
             volume=True if 'volume' in price_data.columns else False)

def plot_entry_exit_with_line(price_data: pd.DataFrame, tradebook: pd.DataFrame, 
                             lookback_days: Optional[int] = None,
                             column: str = 'close') -> None:
    """
    Plot price line chart with entry and exit points.
    
    Args:
        price_data (pd.DataFrame): DataFrame with price data
        tradebook (pd.DataFrame): Tradebook with entry/exit details
        lookback_days (int, optional): Number of days to look back from the last date
        column (str): Column to plot (default: 'close')
    """
    # Convert index to datetime if needed
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data = price_data.copy()
        price_data.index = pd.to_datetime(price_data.index)
    
    # Filter by lookback period if specified
    if lookback_days is not None:
        end_date = price_data.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        price_data = price_data.loc[start_date:end_date]
    
    plt.figure(figsize=(14, 7))
    plt.plot(price_data.index, price_data[column], color='black', linewidth=1)
    
    # Plot long entries and exits
    long_entries = tradebook[tradebook['position'] == 'long']
    for _, trade in long_entries.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        if entry_time in price_data.index:
            plt.scatter(entry_time, trade['entry_price'], marker='^', color='green', s=100)
        
        if exit_time in price_data.index:
            plt.scatter(exit_time, trade['exit_price'], marker='v', color='red', s=100)
    
    # Plot short entries and exits
    short_entries = tradebook[tradebook['position'] == 'short']
    for _, trade in short_entries.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        if entry_time in price_data.index:
            plt.scatter(entry_time, trade['entry_price'], marker='v', color='red', s=100)
        
        if exit_time in price_data.index:
            plt.scatter(exit_time, trade['exit_price'], marker='^', color='green', s=100)
    
    plt.title('Price Chart with Entry and Exit Points')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_performance_summary(metrics: Dict) -> None:
    """
    Plot a performance summary dashboard.
    
    Args:
        metrics (Dict): Dictionary containing performance metrics
    """
    # Define the layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot returns
    axes[0, 0].bar(['Total', 'Annualized'], 
                  [metrics.get('total_return', 0) * 100, metrics.get('annualized_return', 0) * 100],
                  color=['blue', 'green'])
    axes[0, 0].set_title('Returns (%)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0, metrics.get('total_return', 0) * 50, f"{metrics.get('total_return', 0):.2%}", ha='center')
    axes[0, 0].text(1, metrics.get('annualized_return', 0) * 50, f"{metrics.get('annualized_return', 0):.2%}", ha='center')
    
    # Plot win rate
    axes[0, 1].pie([metrics.get('win_rate', 0), 1 - metrics.get('win_rate', 0)], 
                  labels=['Win', 'Loss'], 
                  colors=['green', 'red'],
                  autopct='%1.1f%%',
                  startangle=90)
    axes[0, 1].set_title('Win Rate')
    
    # Plot profit metrics
    profit_metrics = [
        metrics.get('profit_factor', 0),
        metrics.get('sharpe_ratio', 0),
        metrics.get('sortino_ratio', 0)
    ]
    axes[0, 2].bar(['Profit Factor', 'Sharpe', 'Sortino'], profit_metrics, color='purple')
    axes[0, 2].set_title('Profit Metrics')
    axes[0, 2].grid(True, alpha=0.3)
    for i, v in enumerate(profit_metrics):
        axes[0, 2].text(i, v/2, f"{v:.2f}", ha='center')
    
    # Plot average win/loss
    axes[1, 0].bar(['Avg Win', 'Avg Loss'], 
                  [metrics.get('avg_win', 0) * 100, metrics.get('avg_loss', 0) * 100],
                  color=['green', 'red'])
    axes[1, 0].set_title('Avg Win/Loss (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0, metrics.get('avg_win', 0) * 50, f"{metrics.get('avg_win', 0):.2%}", ha='center')
    axes[1, 0].text(1, metrics.get('avg_loss', 0) * 50, f"{metrics.get('avg_loss', 0):.2%}", ha='center')
    
    # Plot max drawdown
    axes[1, 1].bar(['Max Drawdown'], [metrics.get('max_drawdown', 0)], color='red')
    axes[1, 1].set_title('Max Drawdown (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0, metrics.get('max_drawdown', 0)/2, f"{metrics.get('max_drawdown', 0):.2f}%", ha='center')
    
    # Plot trade statistics
    trade_stats = [
        metrics.get('total_trades', 0),
        metrics.get('winning_trades', 0) if 'winning_trades' in metrics else int(metrics.get('total_trades', 0) * metrics.get('win_rate', 0)),
        metrics.get('losing_trades', 0) if 'losing_trades' in metrics else int(metrics.get('total_trades', 0) * (1 - metrics.get('win_rate', 0)))
    ]
    axes[1, 2].bar(['Total', 'Winning', 'Losing'], trade_stats, color=['blue', 'green', 'red'])
    axes[1, 2].set_title('Trade Statistics')
    axes[1, 2].grid(True, alpha=0.3)
    for i, v in enumerate(trade_stats):
        axes[1, 2].text(i, v/2, f"{v}", ha='center')
    
    plt.suptitle('Performance Summary', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_performance_dashboard(equity_curve: pd.DataFrame, 
                               tradebook: pd.DataFrame, 
                               price_data: Optional[pd.DataFrame] = None,
                               metrics: Optional[Dict] = None,
                               benchmark: Optional[pd.DataFrame] = None,
                               lookback_days: Optional[int] = 90) -> None:
    """
    Create a comprehensive performance dashboard.
    
    Args:
        equity_curve (pd.DataFrame): DataFrame with datetime index and 'equity' column
        tradebook (pd.DataFrame): Tradebook with entry/exit details
        price_data (pd.DataFrame, optional): DataFrame with price data
        metrics (Dict, optional): Dictionary containing performance metrics
        benchmark (pd.DataFrame, optional): DataFrame with benchmark data
        lookback_days (int, optional): Number of days to look back for price chart
    """
    # 1. Plot equity curve with optional benchmark
    plot_equity_curve(equity_curve, benchmark)
    
    # 2. Plot drawdown
    plot_drawdown_curve(equity_curve)
    
    # 3. Plot trade analysis
    plot_trade_analysis(tradebook)
    
    # 4. Plot rolling statistics
    plot_rolling_statistics(equity_curve)
    
    # 5. Plot monthly returns heatmap
    plot_monthly_returns_heatmap(equity_curve)
    
    # 6. Plot candlestick with trades if price data is available
    if price_data is not None:
        plot_candlestick_with_trades(price_data, tradebook, lookback_days=lookback_days)
    
    # 7. Plot performance summary if metrics are available
    if metrics is not None:
        plot_performance_summary(metrics) 