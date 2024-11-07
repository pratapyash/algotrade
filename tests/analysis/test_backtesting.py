import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adjust the path to import Backtester and performance module correctly
# This assumes your 'algotrade' directory is in the Python path or you run tests from the project root.
from src.analysis.backtesting import Backtester
from src.analysis import performance 

class TestBacktester(unittest.TestCase):

    def setUp(self):
        """Set up common test data and configurations."""
        self.initial_capital = 100000
        self.commission = 0.001  # 0.1% commission. Renamed from commission_pct
        self.risk_per_trade = 0.01  # 1% risk per trade

        # Create sample market data
        self.dates = pd.to_datetime([datetime(2023, 1, i) for i in range(1, 31)])
        self.data_df = pd.DataFrame({
            'Open': np.random.uniform(90, 110, len(self.dates)),
            'High': np.random.uniform(100, 120, len(self.dates)),
            'Low': np.random.uniform(80, 100, len(self.dates)),
            'Close': np.random.uniform(90, 110, len(self.dates)),
            'Volume': np.random.uniform(1000, 5000, len(self.dates))
        }, index=self.dates)
        # Ensure High is highest and Low is lowest
        self.data_df['High'] = self.data_df[['Open', 'High', 'Low', 'Close']].max(axis=1) + 0.1
        self.data_df['Low'] = self.data_df[['Open', 'High', 'Low', 'Close']].min(axis=1) - 0.1
        self.data_df['Close'] = (self.data_df['High'] + self.data_df['Low']) / 2 # More realistic close

    def _create_simple_strategy_signals(self, signal_type='long', entry_day=5, exit_day=10):
        """Creates a simple strategy signal DataFrame."""
        signals = pd.Series(0, index=self.data_df.index)
        if entry_day < len(self.data_df):
            if signal_type == 'long':
                signals.iloc[entry_day] = 1  # Buy signal
            elif signal_type == 'short':
                signals.iloc[entry_day] = -1 # Short signal
        
        if exit_day < len(self.data_df) and entry_day < exit_day : # Ensure exit is after entry
             # Exit signal can be 0 if SL/TP is used, or specific exit logic
             # For simplicity, we are not setting explicit exit signals here,
             # relying on SL/TP or duration for this basic test.
             # If strategy dictates an exit signal, it would be -1 for long, 1 for short.
             pass
        return signals

    def _create_signal_function(self, signals_series):
        """Creates a signal function from a signals Series."""
        def signal_function(data_df, index):
            if index < len(signals_series):
                return signals_series.iloc[index]
            return 0 # Default to no signal if index is out of bounds
        return signal_function

    def test_initialization(self):
        """Test if the Backtester initializes correctly."""
        # Signals are not passed to init
        bt = Backtester(
            data=self.data_df.copy(),
            initial_capital=self.initial_capital,
            commission=self.commission, # Renamed
            risk_per_trade=self.risk_per_trade,
            stop_loss=0.02, # Renamed
            take_profit=0.04, # Renamed
            compound=True
        )
        self.assertEqual(bt.initial_capital, self.initial_capital)
        self.assertEqual(bt.commission, self.commission)
        self.assertTrue(isinstance(bt.data, pd.DataFrame))

    def test_basic_long_trade_run_and_pnl(self):
        """Test a basic long trade scenario and P&L calculation."""
        entry_day_idx = 5
        exit_day_idx = 10 # Assuming trade lasts this long or hits SL/TP earlier

        signals_series = self._create_simple_strategy_signals(signal_type='long', entry_day=entry_day_idx)
        signal_fn = self._create_signal_function(signals_series)

        bt = Backtester(
            data=self.data_df.copy(),
            initial_capital=self.initial_capital,
            commission=self.commission, # Renamed
            risk_per_trade=self.risk_per_trade,
            stop_loss=0.05, # Renamed
            take_profit=0.10, # Renamed
            compound=True
        )
        bt.run_backtest(signal_function=signal_fn, tearsheet=False) # Changed to run_backtest

        self.assertGreater(len(bt.tradebook), 0, "A trade should have been executed.")
        
        # Check first trade details (if a trade occurred)
        first_trade = bt.tradebook.iloc[0] # Use .iloc for DataFrame
        self.assertEqual(first_trade['type'], 'Long')
        self.assertIsNotNone(first_trade['entry_price'])
        self.assertIsNotNone(first_trade['exit_price'])
        self.assertIsNotNone(first_trade['pnl'])
        
        # P&L check: (Exit Price - Entry Price) * Shares - Commissions
        calculated_pnl = (first_trade['exit_price'] - first_trade['entry_price']) * first_trade['size']
        entry_commission = first_trade['entry_price'] * first_trade['size'] * self.commission
        exit_commission = first_trade['exit_price'] * first_trade['size'] * self.commission
        total_commission = entry_commission + exit_commission
        
        self.assertAlmostEqual(first_trade['pnl'], calculated_pnl - total_commission, places=2)
        self.assertGreater(bt.processed_equity_df.iloc[-1]['equity'], 0)

    def test_no_trades_scenario(self):
        """Test scenario where no trade signals are generated."""
        signals_series = pd.Series(0, index=self.data_df.index) # No signals
        signal_fn = self._create_signal_function(signals_series)

        bt = Backtester(
            data=self.data_df.copy(),
            initial_capital=self.initial_capital,
            commission=self.commission, # Renamed
            risk_per_trade=self.risk_per_trade,
            stop_loss=0.02, # Renamed
            take_profit=0.04 # Renamed
        )
        bt.run_backtest(signal_function=signal_fn, tearsheet=False) # Changed to run_backtest
        self.assertEqual(len(bt.tradebook), 0, "No trades should have occurred.")
        self.assertEqual(bt.processed_equity_df.iloc[-1]['equity'], self.initial_capital, 
                         "Equity should remain initial capital if no trades.")

    def test_commission_application(self):
        """Test if commissions are applied correctly."""
        entry_day_idx = 3
        signals_series = self._create_simple_strategy_signals(signal_type='long', entry_day=entry_day_idx)
        signal_fn = self._create_signal_function(signals_series)

        bt = Backtester(
            data=self.data_df.copy(),
            initial_capital=self.initial_capital,
            commission=self.commission, # Use the setup commission
            risk_per_trade=0.01,
            stop_loss=0.05, # Renamed
            take_profit=0.10, # Renamed
            compound=False # Non-compounding for simpler P&L verification initially
        )
        bt.run_backtest(signal_function=signal_fn, tearsheet=False) # Changed to run_backtest

        if bt.tradebook.empty:
            return 

        first_trade = bt.tradebook.iloc[0]
        
        expected_entry_commission = first_trade['entry_price'] * first_trade['size'] * self.commission
        expected_exit_commission = first_trade['exit_price'] * first_trade['size'] * self.commission
        expected_total_commission = expected_entry_commission + expected_exit_commission
        
        # Gross PnL = (Exit Price - Entry Price) * Shares
        gross_pnl = (first_trade['exit_price'] - first_trade['entry_price']) * first_trade['size']
        
        self.assertAlmostEqual(first_trade['pnl'], gross_pnl - expected_total_commission, places=2)
        self.assertAlmostEqual(bt.current_capital, self.initial_capital + first_trade['pnl'], places=2)

    def test_stop_loss_trigger(self):
        """Test if stop-loss is triggered correctly for a long trade."""
        entry_day_idx = 2
        sl_pct = 0.05
        
        signals_series = pd.Series(0, index=self.data_df.index)
        signals_series.iloc[entry_day_idx] = 1 # Long signal
        signal_fn = self._create_signal_function(signals_series)

        data_copy = self.data_df.copy()
        entry_price_estimate = data_copy['Close'].iloc[entry_day_idx] # Entry on close of signal day
        stop_loss_level = entry_price_estimate * (1 - sl_pct)

        if entry_day_idx + 1 < len(data_copy):
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'Low'] = stop_loss_level * 0.99 # Ensure it triggers
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'Open'] = stop_loss_level * 1.02
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'High'] = stop_loss_level * 1.03
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'Close'] = stop_loss_level * 1.01
        else:
            self.skipTest("Not enough data to test SL trigger after entry.")
            return

        bt = Backtester(
            data=data_copy,
            initial_capital=self.initial_capital,
            commission=0, # No commission for simplicity
            risk_per_trade=0.01,
            stop_loss=sl_pct, # Renamed
            take_profit=0.10, # TP far away, Renamed
        )
        bt.run_backtest(signal_function=signal_fn, tearsheet=False) # Changed to run_backtest

        self.assertTrue(len(bt.tradebook) > 0, "A trade should have been made.")
        first_trade = bt.tradebook.iloc[0]
        self.assertEqual(first_trade['exit_reason'], 'Stop Loss') # Assuming exit_reason is populated
        self.assertAlmostEqual(first_trade['exit_price'], first_trade['entry_price'] * (1 - sl_pct), places=5)

    def test_take_profit_trigger(self):
        """Test if take-profit is triggered correctly for a long trade."""
        entry_day_idx = 2
        tp_pct = 0.05

        signals_series = pd.Series(0, index=self.data_df.index)
        signals_series.iloc[entry_day_idx] = 1 # Long signal
        signal_fn = self._create_signal_function(signals_series)

        data_copy = self.data_df.copy()
        entry_price_estimate = data_copy['Close'].iloc[entry_day_idx]
        take_profit_level = entry_price_estimate * (1 + tp_pct)

        if entry_day_idx + 1 < len(data_copy):
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'High'] = take_profit_level * 1.01 # Ensure TP hit
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'Open'] = take_profit_level * 0.98
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'Low'] = take_profit_level * 0.97
            data_copy.loc[data_copy.index[entry_day_idx + 1], 'Close'] = take_profit_level * 0.99
        else:
            self.skipTest("Not enough data to test TP trigger after entry.")
            return

        bt = Backtester(
            data=data_copy,
            initial_capital=self.initial_capital,
            commission=0, # No commission
            risk_per_trade=0.01,
            stop_loss=0.10, # SL far away, Renamed
            take_profit=tp_pct, # Renamed
        )
        bt.run_backtest(signal_function=signal_fn, tearsheet=False) # Changed to run_backtest

        self.assertTrue(len(bt.tradebook) > 0, "A trade should have been made for TP test.")
        first_trade = bt.tradebook.iloc[0]
        self.assertEqual(first_trade['exit_reason'], 'Take Profit') # Assuming exit_reason is populated
        self.assertAlmostEqual(first_trade['exit_price'], first_trade['entry_price'] * (1 + tp_pct), places=5)

    def test_compounding_effect(self):
        """Test difference in position sizing and P&L with compounding."""
        signals_series = self._create_simple_strategy_signals(signal_type='long', entry_day=2, exit_day=8)
        signal_fn = self._create_signal_function(signals_series)
        
        # Run with compounding
        bt_compound = Backtester(
            data=self.data_df.copy(), initial_capital=self.initial_capital,
            commission=0, risk_per_trade=0.02, stop_loss=0.05, take_profit=0.1, compound=True # Renamed params
        )
        bt_compound.run_backtest(signal_function=signal_fn, tearsheet=False)

        # Run without compounding
        bt_no_compound = Backtester(
            data=self.data_df.copy(), initial_capital=self.initial_capital,
            commission=0, risk_per_trade=0.02, stop_loss=0.05, take_profit=0.1, compound=False # Renamed params
        )
        bt_no_compound.run_backtest(signal_function=signal_fn, tearsheet=False)

        if bt_compound.tradebook.empty or bt_no_compound.tradebook.empty:
            self.skipTest("Not enough trades to compare compounding effects. Adjust data or signals.")
            return

        self.assertTrue(len(bt_compound.tradebook) > 0)
        self.assertTrue(len(bt_no_compound.tradebook) > 0)
        self.assertNotEqual(bt_compound.current_capital, bt_no_compound.current_capital, 
                            "Final capital should differ if trades occurred and compounding was different, unless PnL was zero.")

    def test_short_trade_pnl(self):
        """Test a basic short trade scenario and P&L calculation."""
        entry_day_idx = 5
        signals_series = self._create_simple_strategy_signals(signal_type='short', entry_day=entry_day_idx)
        signal_fn = self._create_signal_function(signals_series)
        
        bt = Backtester(
            data=self.data_df.copy(),
            initial_capital=self.initial_capital,
            commission=self.commission, # Renamed
            risk_per_trade=self.risk_per_trade,
            stop_loss=0.05, # Renamed
            take_profit=0.10, # Renamed
            compound=True
        )
        bt.run_backtest(signal_function=signal_fn, tearsheet=False) # Changed to run_backtest

        if bt.tradebook.empty:
            self.skipTest("No short trade executed. Adjust data or signals for short trade test.")
            return

        first_trade = bt.tradebook.iloc[0]
        self.assertEqual(first_trade['type'], 'Short')
        self.assertIsNotNone(first_trade['entry_price'])
        self.assertIsNotNone(first_trade['exit_price'])
        
        # P&L for short: (Entry Price - Exit Price) * Shares - Commissions
        calculated_pnl = (first_trade['entry_price'] - first_trade['exit_price']) * first_trade['size']
        entry_commission = first_trade['entry_price'] * first_trade['size'] * self.commission
        exit_commission = first_trade['exit_price'] * first_trade['size'] * self.commission
        total_commission = entry_commission + exit_commission
        
        self.assertAlmostEqual(first_trade['pnl'], calculated_pnl - total_commission, places=2)

    # TODO: Add more tests:
    # - Risk per trade enforcement (check position size calculation more directly)
    # - Performance metrics calculation (Sharpe, Sortino, Max Drawdown) - requires known results
    # - Edge case: Signal on the last day of data
    # - Edge case: Data with insufficient length for a trade to complete
    # - Test with `trade_on_close=False` (i.e. trade on next bar's open) if that feature is distinct

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

