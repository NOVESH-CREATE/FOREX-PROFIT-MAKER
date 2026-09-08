"""Pure ORB Scenario-3 backtest engine, extracted verbatim from app.py (commit 63e5213)."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings("ignore")


# ================ SCENARIO_3_SETUPS (verbatim from app.py) ================
SCENARIO_3_SETUPS = {
    'Monday': [
        {'pair': 'USDCAD', 'time_ist': '09:15 PM', 'time_gmt': '15:45', 'entry_mode': '15min', 'rr': 3.0, 'expected_wr': 61.5}
    ],
    'Tuesday': [],  # No trades - below 60% win rate
    'Wednesday': [
        {'pair': 'GBPUSD', 'time_ist': '08:30 PM', 'time_gmt': '15:00', 'entry_mode': '15min', 'rr': 3.0, 'expected_wr': 61.5}
    ],
    'Thursday': [
        {'pair': 'USDCAD', 'time_ist': '04:45 PM', 'time_gmt': '11:15', 'entry_mode': '15min', 'rr': 2.0, 'expected_wr': 72.7},
        {'pair': 'EURUSD', 'time_ist': '05:00 PM', 'time_gmt': '11:30', 'entry_mode': '5min', 'rr': 2.0, 'expected_wr': 66.7},
        {'pair': 'GBPUSD', 'time_ist': '09:15 AM', 'time_gmt': '03:45', 'entry_mode': '15min', 'rr': 2.0, 'expected_wr': 63.6}
    ],
    'Friday': [
        {'pair': 'USDCAD', 'time_ist': '09:30 PM', 'time_gmt': '16:00', 'entry_mode': '15min', 'rr': 2.5, 'expected_wr': 63.0}
    ]
}


# ================ RealTradesCompounding (verbatim from app.py) ================
class RealTradesCompounding:
    """
    Apply compounding to REAL backtest trades only.
    NO simulation - uses actual historical data.
    """
    
    def __init__(self, initial_capital: float, risk_percent: float = 10.0):
        self.initial_capital = initial_capital
        self.risk_percent = risk_percent
    
    def get_fixed_risk(self, balance: float) -> float:
        """Fixed risk - always risk same amount based on initial capital"""
        return self.initial_capital * (self.risk_percent / 100)
    
    def get_full_compound_risk(self, balance: float) -> float:
        """Full compounding - always risk percentage of current balance"""
        return balance * (self.risk_percent / 100)
    
    def get_milestone_risk(self, balance: float) -> float:
        """Milestone compounding - risk increases at certain balance levels"""
        ic = self.initial_capital
        
        # Define milestones
        if balance >= ic * 20:
            return ic * 2.00
        elif balance >= ic * 15:
            return ic * 1.50
        elif balance >= ic * 10:
            return ic * 1.00
        elif balance >= ic * 6:
            return ic * 0.60
        elif balance >= ic * 4:
            return ic * 0.40
        elif balance >= ic * 3:
            return ic * 0.30
        elif balance >= ic * 2:
            return ic * 0.20
        elif balance >= ic * 1.5:
            return ic * 0.15
        else:
            return ic * 0.10
    
    def get_milestone_table(self) -> list:
        """Get milestone table for display"""
        ic = self.initial_capital
        
        return [
            {"balance_range": f"${ic:.0f} - ${ic*1.5-0.01:.0f}", "risk": ic * 0.10},
            {"balance_range": f"${ic*1.5:.0f} - ${ic*2-0.01:.0f}", "risk": ic * 0.15},
            {"balance_range": f"${ic*2:.0f} - ${ic*3-0.01:.0f}", "risk": ic * 0.20},
            {"balance_range": f"${ic*3:.0f} - ${ic*4-0.01:.0f}", "risk": ic * 0.30},
            {"balance_range": f"${ic*4:.0f} - ${ic*6-0.01:.0f}", "risk": ic * 0.40},
            {"balance_range": f"${ic*6:.0f} - ${ic*10-0.01:.0f}", "risk": ic * 0.60},
            {"balance_range": f"${ic*10:.0f} - ${ic*15-0.01:.0f}", "risk": ic * 1.00},
            {"balance_range": f"${ic*15:.0f} - ${ic*20-0.01:.0f}", "risk": ic * 1.50},
            {"balance_range": f"${ic*20:.0f}+", "risk": ic * 2.00},
        ]
    
    def apply_to_real_trades(self, trades_df: pd.DataFrame, method: str = "milestone") -> dict:
        """
        Apply compounding to REAL backtest trades.
        This uses ACTUAL historical trades, not simulations.
        
        Parameters:
        - trades_df: DataFrame with real backtest results
        - method: "fixed", "milestone", or "full_compound"
        
        Returns:
        - Dictionary with equity curve and statistics
        """
        
        if trades_df.empty:
            return None
        
        balance = self.initial_capital
        equity_curve = [balance]
        trade_details = []
        
        peak_balance = balance
        max_drawdown = 0
        max_drawdown_pct = 0
        
        # Sort trades chronologically
        if 'entry_time' in trades_df.columns:
            trades_sorted = trades_df.sort_values('entry_time').reset_index(drop=True)
        else:
            trades_sorted = trades_df.sort_values('date').reset_index(drop=True)
        
        for idx, trade in trades_sorted.iterrows():
            # Get risk based on current balance and method
            if method == "fixed":
                risk = self.get_fixed_risk(balance)
            elif method == "full_compound":
                risk = self.get_full_compound_risk(balance)
            else:  # milestone
                risk = self.get_milestone_risk(balance)
            
            # Safety: don't risk more than 50% of current balance
            risk = min(risk, balance * 0.5)
            
            # Get ACTUAL RR from this specific trade
            trade_rr = trade['rr_ratio']
            
            # Calculate P&L based on REAL result
            if trade['result'] == 'WIN':
                pnl = risk * trade_rr
            else:
                pnl = -risk
            
            # Update balance
            old_balance = balance
            balance += pnl
            equity_curve.append(balance)
            
            # Track peak and drawdown
            if balance > peak_balance:
                peak_balance = balance
            
            current_dd = peak_balance - balance
            current_dd_pct = (current_dd / peak_balance * 100) if peak_balance > 0 else 0
            
            if current_dd > max_drawdown:
                max_drawdown = current_dd
            if current_dd_pct > max_drawdown_pct:
                max_drawdown_pct = current_dd_pct
            
            # Store trade details
            trade_details.append({
                'trade_num': idx + 1,
                'date': trade['date'],
                'day': trade['day_of_week'],
                'pair': trade['pair'],
                'time_ist': trade.get('orb_time_ist', ''),
                'direction': trade.get('direction', ''),
                'rr': trade_rr,
                'result': trade['result'],
                'risk_used': round(risk, 2),
                'pnl': round(pnl, 2),
                'balance_before': round(old_balance, 2),
                'balance_after': round(balance, 2),
                'drawdown_pct': round(current_dd_pct, 2)
            })
            
            # Stop if account blown
            if balance <= 0:
                break
        
        # Calculate final statistics
        total_trades = len(trade_details)
        wins = sum(1 for t in trade_details if t['result'] == 'WIN')
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = balance - self.initial_capital
        total_return_pct = (total_profit / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in trade_details if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trade_details if t['pnl'] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        return {
            'method': method,
            'initial_capital': self.initial_capital,
            'final_balance': round(balance, 2),
            'total_profit': round(total_profit, 2),
            'total_return_pct': round(total_return_pct, 2),
            'peak_balance': round(peak_balance, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'equity_curve': equity_curve,
            'trade_details': trade_details
        }
    
    def compare_all_methods(self, trades_df: pd.DataFrame) -> dict:
        """
        Apply all 3 compounding methods to the same real trades.
        Returns comparison of all methods.
        """
        
        results = {}
        
        for method in ["fixed", "milestone", "full_compound"]:
            result = self.apply_to_real_trades(trades_df, method)
            if result:
                results[method] = result
        
        return results


# ================ parse_mt5_csv (verbatim from app.py) ================
def parse_mt5_csv(content, pair_name=""):
    """Parse MT5 exported CSV/TXT data"""
    lines = content.strip().split('\n')
    data = []
    
    for line in lines:
        if any(header in line.upper() for header in ['DATE', 'TIME', 'OPEN', '<DATE>', '<TIME>']):
            continue
        if not line.strip():
            continue
        
        if '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 6:
            try:
                date_str = parts[0]
                time_str = parts[1]
                
                if '.' in date_str:
                    date_format = "%Y.%m.%d"
                elif '-' in date_str:
                    date_format = "%Y-%m-%d"
                elif '/' in date_str:
                    date_format = "%Y/%m/%d"
                else:
                    continue
                
                datetime_str = f"{date_str} {time_str}"
                
                try:
                    dt = datetime.strptime(datetime_str, f"{date_format} %H:%M:%S")
                except:
                    try:
                        dt = datetime.strptime(datetime_str, f"{date_format} %H:%M")
                    except:
                        continue
                
                open_price = float(parts[2])
                high_price = float(parts[3])
                low_price = float(parts[4])
                close_price = float(parts[5])
                
                data.append({
                    'datetime': dt,
                    'date': dt.date(),
                    'time': dt.time(),
                    'time_str': dt.strftime('%H:%M'),
                    'day_of_week': dt.strftime('%A'),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price
                })
                
            except Exception as e:
                continue
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime'], keep='first')
    
    return df


# ================ get_pip_value (verbatim from app.py) ================
def get_pip_value(pair):
    """Get pip value for different currency pairs"""
    pair = pair.upper().replace('/', '').replace(' ', '')
    if 'JPY' in pair:
        return 0.01
    else:
        return 0.0001


# ================ simulate_trade (verbatim from app.py) ================
def simulate_trade(subsequent_candles, entry, sl, tp, direction, pip_value):
    """Simulate a single trade outcome"""
    for idx, candle in subsequent_candles.iterrows():
        if direction == 'LONG':
            if candle['low'] <= sl:
                return {
                    'result': 'LOSS',
                    'pnl_pips': round((sl - entry) / pip_value, 1),
                    'exit_time': candle['datetime'],
                    'exit_price': sl,
                    'exit_reason': 'SL Hit'
                }
            if candle['high'] >= tp:
                return {
                    'result': 'WIN',
                    'pnl_pips': round((tp - entry) / pip_value, 1),
                    'exit_time': candle['datetime'],
                    'exit_price': tp,
                    'exit_reason': 'TP Hit'
                }
        else:  # SHORT
            if candle['high'] >= sl:
                return {
                    'result': 'LOSS',
                    'pnl_pips': round((entry - sl) / pip_value, 1),
                    'exit_time': candle['datetime'],
                    'exit_price': sl,
                    'exit_reason': 'SL Hit'
                }
            if candle['low'] <= tp:
                return {
                    'result': 'WIN',
                    'pnl_pips': round((entry - tp) / pip_value, 1),
                    'exit_time': candle['datetime'],
                    'exit_price': tp,
                    'exit_reason': 'TP Hit'
                }
    
    # End of day - close at last price
    if not subsequent_candles.empty:
        last = subsequent_candles.iloc[-1]
        if direction == 'LONG':
            pnl = (last['close'] - entry) / pip_value
        else:
            pnl = (entry - last['close']) / pip_value
        
        return {
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'pnl_pips': round(pnl, 1),
            'exit_time': last['datetime'],
            'exit_price': last['close'],
            'exit_reason': 'EOD Close'
        }
    
    return {
        'result': 'NO_TRADE',
        'pnl_pips': 0,
        'exit_time': None,
        'exit_price': None,
        'exit_reason': 'No Data'
    }


# ================ execute_setup (verbatim from app.py) ================
def execute_setup(setup, df_5min, df_15min, date, buffer_pips=2):
    """Execute a single trading setup on historical data"""
    pair = setup['pair']
    orb_time_gmt = setup['time_gmt']
    entry_mode = setup['entry_mode']
    rr_ratio = setup['rr']
    pip_value = get_pip_value(pair)
    buffer = buffer_pips * pip_value
    
    # Find the ORB candle
    orb_candle = df_15min[
        (df_15min['date'] == date) & 
        (df_15min['time_str'] == orb_time_gmt)
    ]
    
    if orb_candle.empty:
        return None
    
    orb = orb_candle.iloc[0]
    orb_high = orb['high']
    orb_low = orb['low']
    orb_datetime = orb['datetime']
    orb_range_pips = (orb_high - orb_low) / pip_value
    
    # Filter invalid ORB ranges
    if orb_range_pips < 2 or orb_range_pips > 100:
        return None
    
    # Get candles after ORB for breakout detection
    if entry_mode == '5min':
        orb_end_time = orb_datetime + timedelta(minutes=15)
        day_candles = df_5min[
            (df_5min['date'] == date) & 
            (df_5min['datetime'] >= orb_end_time)
        ].sort_values('datetime')
        breakout_df = df_5min
    else:  # 15min
        day_candles = df_15min[
            (df_15min['date'] == date) & 
            (df_15min['datetime'] > orb_datetime)
        ].sort_values('datetime')
        breakout_df = df_15min
    
    # Look for breakout
    for _, candle in day_candles.iterrows():
        # LONG breakout
        if candle['close'] > orb_high and candle['high'] > orb_high:
            entry_price = candle['close']
            sl_price = orb_low - buffer
            risk = entry_price - sl_price
            
            if risk <= 0:
                continue
            
            tp_price = entry_price + (risk * rr_ratio)
            
            subsequent = breakout_df[
                (breakout_df['date'] == date) & 
                (breakout_df['datetime'] > candle['datetime'])
            ].sort_values('datetime')
            
            result = simulate_trade(subsequent, entry_price, sl_price, tp_price, 'LONG', pip_value)
            
            return {
                'date': date,
                'day_of_week': orb['day_of_week'],
                'pair': pair,
                'orb_time_gmt': orb_time_gmt,
                'orb_time_ist': setup['time_ist'],
                'entry_mode': entry_mode,
                'rr_ratio': rr_ratio,
                'expected_wr': setup.get('expected_wr', 0),
                'direction': 'LONG',
                'orb_high': round(orb_high, 5),
                'orb_low': round(orb_low, 5),
                'orb_range_pips': round(orb_range_pips, 1),
                'entry_time': candle['datetime'],
                'entry_price': round(entry_price, 5),
                'sl_price': round(sl_price, 5),
                'tp_price': round(tp_price, 5),
                'risk_pips': round(risk / pip_value, 1),
                'result': result['result'],
                'pnl_pips': result['pnl_pips'],
                'exit_time': result['exit_time'],
                'exit_price': result['exit_price'],
                'exit_reason': result['exit_reason']
            }
        
        # SHORT breakout
        elif candle['close'] < orb_low and candle['low'] < orb_low:
            entry_price = candle['close']
            sl_price = orb_high + buffer
            risk = sl_price - entry_price
            
            if risk <= 0:
                continue
            
            tp_price = entry_price - (risk * rr_ratio)
            
            subsequent = breakout_df[
                (breakout_df['date'] == date) & 
                (breakout_df['datetime'] > candle['datetime'])
            ].sort_values('datetime')
            
            result = simulate_trade(subsequent, entry_price, sl_price, tp_price, 'SHORT', pip_value)
            
            return {
                'date': date,
                'day_of_week': orb['day_of_week'],
                'pair': pair,
                'orb_time_gmt': orb_time_gmt,
                'orb_time_ist': setup['time_ist'],
                'entry_mode': entry_mode,
                'rr_ratio': rr_ratio,
                'expected_wr': setup.get('expected_wr', 0),
                'direction': 'SHORT',
                'orb_high': round(orb_high, 5),
                'orb_low': round(orb_low, 5),
                'orb_range_pips': round(orb_range_pips, 1),
                'entry_time': candle['datetime'],
                'entry_price': round(entry_price, 5),
                'sl_price': round(sl_price, 5),
                'tp_price': round(tp_price, 5),
                'risk_pips': round(risk / pip_value, 1),
                'result': result['result'],
                'pnl_pips': result['pnl_pips'],
                'exit_time': result['exit_time'],
                'exit_price': result['exit_price'],
                'exit_reason': result['exit_reason']
            }
    
    return None


# ================ backtest_scenario_3 (verbatim from app.py) ================
def backtest_scenario_3(pairs_data, buffer_pips=2):
    """
    Backtest ONLY Scenario 3 (60%+ Win Rate trades).
    Uses REAL historical data from uploaded CSV files.
    """
    
    all_trades = []
    
    # Get all available dates
    all_dates = set()
    for pair, data in pairs_data.items():
        if 'm15' in data:
            all_dates.update(data['m15']['date'].unique())
    
    all_dates = sorted(all_dates)
    
    # Process each date
    for date in all_dates:
        day_of_week = date.strftime('%A')
        
        # Skip weekends
        if day_of_week in ['Saturday', 'Sunday']:
            continue
        
        # Skip days not in scenario
        if day_of_week not in SCENARIO_3_SETUPS:
            continue
        
        day_setups = SCENARIO_3_SETUPS[day_of_week]
        
        # Execute each setup for this day
        for setup in day_setups:
            pair = setup['pair'].upper().replace('/', '').replace(' ', '')
            
            if pair not in pairs_data:
                continue
            
            pair_data = pairs_data[pair]
            
            if 'm5' not in pair_data or 'm15' not in pair_data:
                continue
            
            trade = execute_setup(
                setup, 
                pair_data['m5'], 
                pair_data['m15'], 
                date, 
                buffer_pips
            )
            
            if trade and trade['result'] != 'NO_TRADE':
                all_trades.append(trade)
    
    return pd.DataFrame(all_trades)


# ================ calculate_backtest_statistics (verbatim from app.py) ================
def calculate_backtest_statistics(trades_df):
    """Calculate comprehensive backtest statistics"""
    
    if trades_df.empty:
        return None
    
    total = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Day-wise statistics
    day_stats = trades_df.groupby('day_of_week').apply(
        lambda x: pd.Series({
            'trades': len(x),
            'wins': (x['result'] == 'WIN').sum(),
            'losses': (x['result'] == 'LOSS').sum(),
            'win_rate': (x['result'] == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
        })
    )
    
    # Pair-wise statistics
    pair_stats = trades_df.groupby('pair').apply(
        lambda x: pd.Series({
            'trades': len(x),
            'wins': (x['result'] == 'WIN').sum(),
            'losses': (x['result'] == 'LOSS').sum(),
            'win_rate': (x['result'] == 'WIN').sum() / len(x) * 100 if len(x) > 0 else 0
        })
    )
    
    # Consecutive wins/losses
    results = trades_df['result'].tolist()
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for r in results:
        if r == 'WIN':
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_rr': round(trades_df['rr_ratio'].mean(), 2),
        'day_stats': day_stats,
        'pair_stats': pair_stats
    }

