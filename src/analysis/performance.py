import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union

def calculate_cumulative_pnl(tradebook: pd.DataFrame, 
                           initial_capital: float, 
                           transaction_cost: float) -> pd.DataFrame:
    """
    Calculate cumulative PnL for trades.
    
    Args:
        tradebook (pd.DataFrame): Tradebook containing trades
        initial_capital (float): Initial capital
        transaction_cost (float): Transaction cost per trade
        
    Returns:
        pd.DataFrame: Tradebook with PnL calculations
    """
    tradebook['pnl'] = 0.0
    tradebook['cum_pnl'] = 0.0
    tradebook['capital'] = initial_capital
    
    for i in range(len(tradebook)):
        entry_price = tradebook.iloc[i]['entry_price']
        exit_price = tradebook.iloc[i]['exit_price']
        trade_type = tradebook.iloc[i]['position']
        
        if trade_type == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
            
        # Apply transaction costs
        pnl -= 2 * transaction_cost  # Entry and exit costs
        
        tradebook.loc[tradebook.index[i], 'pnl'] = pnl
        
        if i == 0:
            tradebook.loc[tradebook.index[i], 'cum_pnl'] = pnl
            tradebook.loc[tradebook.index[i], 'capital'] = initial_capital * (1 + pnl)
        else:
            prev_cum_pnl = tradebook.iloc[i-1]['cum_pnl']
            tradebook.loc[tradebook.index[i], 'cum_pnl'] = prev_cum_pnl + pnl
            tradebook.loc[tradebook.index[i], 'capital'] = initial_capital * (1 + prev_cum_pnl + pnl)
    
    return tradebook

def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series for equity curve.
    
    Args:
        equity_curve (pd.Series): Equity curve series
        
    Returns:
        pd.Series: Drawdown series in percentage
    """
    # Calculate rolling maximum
    rolling_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    return drawdown

def calculate_runup_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate runup series for equity curve.
    
    Args:
        equity_curve (pd.Series): Equity curve series
        
    Returns:
        pd.Series: Runup series in percentage
    """
    # Calculate rolling minimum
    rolling_min = equity_curve.cummin()
    
    # Calculate runup
    runup = (equity_curve - rolling_min) / rolling_min * 100
    
    return runup

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns (pd.Series): Daily returns series
        risk_free_rate (float): Risk-free rate (annualized)
        annualization_factor (int): Annualization factor (252 for daily, 12 for monthly, 4 for quarterly)
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / annualization_factor)
    sharpe = excess_returns.mean() / returns.std() * (annualization_factor ** 0.5)
    
    return sharpe

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sortino ratio (only considers downside deviation).
    
    Args:
        returns (pd.Series): Daily returns series
        risk_free_rate (float): Risk-free rate (annualized)
        annualization_factor (int): Annualization factor (252 for daily, 12 for monthly, 4 for quarterly)
        
    Returns:
        float: Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / annualization_factor)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    sortino = excess_returns.mean() / downside_returns.std() * (annualization_factor ** 0.5)
    
    return sortino

def calculate_max_drawdown(drawdown_series: pd.Series) -> float:
    """
    Calculate maximum drawdown from a drawdown series.
    
    Args:
        drawdown_series (pd.Series): Drawdown series in percentage
        
    Returns:
        float: Maximum drawdown in percentage
    """
    return drawdown_series.min() if len(drawdown_series) > 0 else 0.0

def calculate_metrics(tradebook: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive performance metrics for the strategy.
    
    Args:
        tradebook (pd.DataFrame): Tradebook with PnL calculations
        equity_curve (pd.DataFrame): Equity curve with datetime index
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Trade statistics
    total_trades = len(tradebook)
    winning_trades = len(tradebook[tradebook['pnl'] > 0])
    losing_trades = len(tradebook[tradebook['pnl'] < 0])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = tradebook[tradebook['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = tradebook[tradebook['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    # Profit metrics
    gross_profit = tradebook[tradebook['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
    gross_loss = tradebook[tradebook['pnl'] < 0]['pnl'].sum() if losing_trades > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    # Return metrics
    total_return = tradebook['cum_pnl'].iloc[-1] if len(tradebook) > 0 else 0
    
    # Calculate daily returns
    daily_returns = equity_curve['equity'].pct_change().dropna()
    
    # Calculate annualized return
    if len(daily_returns) > 1:
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    else:
        annualized_return = 0
    
    # Calculate risk metrics
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    sortino_ratio = calculate_sortino_ratio(daily_returns)
    
    # Calculate drawdown if not already in equity_curve
    if 'drawdown' in equity_curve.columns:
        drawdown_series = equity_curve['drawdown']
    else:
        drawdown_series = calculate_drawdown_series(equity_curve['equity'])
    
    max_drawdown = calculate_max_drawdown(drawdown_series)
    
    # Calculate holding durations
    if 'entry_time' in tradebook.columns and 'exit_time' in tradebook.columns:
        durations = pd.to_datetime(tradebook['exit_time']) - pd.to_datetime(tradebook['entry_time'])
        avg_holding_duration = durations.mean() if len(durations) > 0 else pd.Timedelta(0)
    else:
        avg_holding_duration = pd.Timedelta(0)
    
    # Calculate largest win/loss
    largest_win = tradebook['pnl'].max() if len(tradebook) > 0 else 0
    largest_loss = tradebook['pnl'].min() if len(tradebook) > 0 else 0
    
    # Calculate buy & hold return
    if len(equity_curve) > 1:
        price_start = equity_curve['equity'].iloc[0]
        price_end = equity_curve['equity'].iloc[-1]
        buy_and_hold_return = (price_end - price_start) / price_start
    else:
        buy_and_hold_return = 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'avg_holding_duration': avg_holding_duration,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'buy_and_hold_return': buy_and_hold_return
    }

def analyze_trade_distribution(tradebook: pd.DataFrame) -> Dict:
    """
    Analyze trade distribution by various factors.
    
    Args:
        tradebook (pd.DataFrame): Tradebook with PnL calculations
        
    Returns:
        dict: Dictionary containing trade distribution analysis
    """
    # Position type analysis
    if 'position' in tradebook.columns:
        long_trades = tradebook[tradebook['position'] == 'long']
        short_trades = tradebook[tradebook['position'] == 'short']
        
        long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        long_avg_return = long_trades['pnl'].mean() if len(long_trades) > 0 else 0
        short_avg_return = short_trades['pnl'].mean() if len(short_trades) > 0 else 0
    else:
        long_win_rate = 0
        short_win_rate = 0
        long_avg_return = 0
        short_avg_return = 0
    
    # Time analysis if entry_time is available
    day_of_week_performance = {}
    month_performance = {}
    
    if 'entry_time' in tradebook.columns:
        tradebook = tradebook.copy()
        tradebook['entry_time'] = pd.to_datetime(tradebook['entry_time'])
        
        # Day of week analysis
        tradebook['day_of_week'] = tradebook['entry_time'].dt.day_name()
        day_groups = tradebook.groupby('day_of_week')
        
        for day, group in day_groups:
            day_of_week_performance[day] = {
                'count': len(group),
                'win_rate': len(group[group['pnl'] > 0]) / len(group) if len(group) > 0 else 0,
                'avg_return': group['pnl'].mean() if len(group) > 0 else 0
            }
        
        # Month analysis
        tradebook['month'] = tradebook['entry_time'].dt.month_name()
        month_groups = tradebook.groupby('month')
        
        for month, group in month_groups:
            month_performance[month] = {
                'count': len(group),
                'win_rate': len(group[group['pnl'] > 0]) / len(group) if len(group) > 0 else 0,
                'avg_return': group['pnl'].mean() if len(group) > 0 else 0
            }
    
    return {
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'long_avg_return': long_avg_return,
        'short_avg_return': short_avg_return,
        'day_of_week_performance': day_of_week_performance,
        'month_performance': month_performance
    }

def print_performance_summary(metrics: Dict) -> None:
    """
    Print a formatted performance summary.
    
    Args:
        metrics (Dict): Dictionary containing performance metrics
    """
    print("\n" + "="*50)
    print(" "*15 + "PERFORMANCE SUMMARY")
    print("="*50)
    
    # Returns
    print("\n-- RETURNS --")
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"Buy & Hold Return: {metrics.get('buy_and_hold_return', 0):.2%}")
    
    # Trade Statistics
    print("\n-- TRADE STATISTICS --")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Winning Trades: {metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0):.2%})")
    print(f"Losing Trades: {metrics.get('losing_trades', 0)} ({1 - metrics.get('win_rate', 0):.2%})")
    print(f"Average Win: {metrics.get('avg_win', 0):.2%}")
    print(f"Average Loss: {metrics.get('avg_loss', 0):.2%}")
    print(f"Largest Win: {metrics.get('largest_win', 0):.2%}")
    print(f"Largest Loss: {metrics.get('largest_loss', 0):.2%}")
    print(f"Average Holding Duration: {metrics.get('avg_holding_duration', 0)}")
    
    # Risk Metrics
    print("\n-- RISK METRICS --")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    
    print("\n" + "="*50) 