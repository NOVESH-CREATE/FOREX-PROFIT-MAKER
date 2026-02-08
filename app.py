import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Try to import reportlab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

st.set_page_config(
    page_title="ORB Backtester + Compounding",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .profit-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .loss-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .neutral-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .milestone-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .compound-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .winner-banner {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SCENARIO 3 SETUPS (60%+ Win Rate Only) ====================

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


# ==================== COMPOUNDING CALCULATOR (REAL TRADES ONLY) ====================

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


# ==================== UTILITY FUNCTIONS ====================

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


def get_pip_value(pair):
    """Get pip value for different currency pairs"""
    pair = pair.upper().replace('/', '').replace(' ', '')
    if 'JPY' in pair:
        return 0.01
    else:
        return 0.0001


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


# ==================== STREAMLIT UI ====================

# Initialize session state
if 'backtest_trades' not in st.session_state:
    st.session_state.backtest_trades = None
if 'backtest_stats' not in st.session_state:
    st.session_state.backtest_stats = None
if 'compounding_results' not in st.session_state:
    st.session_state.compounding_results = None

# Title
st.title("📊 ORB Backtester + Compounding Calculator")
st.markdown("### Scenario 3: 60%+ Win Rate Trades Only")
st.markdown("*Uses REAL historical data from your CSV files - No simulations!*")

st.markdown("---")

# ==================== SIDEBAR ====================

st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Backtest Settings")
buffer_pips = st.sidebar.slider("SL Buffer (pips)", 0, 10, 2)

st.sidebar.subheader("Compounding Settings")
initial_capital = st.sidebar.number_input(
    "Starting Capital ($)", 
    min_value=10.0, 
    max_value=100000.0, 
    value=50.0, 
    step=10.0
)
risk_percent = st.sidebar.slider("Base Risk %", 1, 20, 10)


# ==================== MAIN CONTENT ====================

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Upload & Backtest", "💰 Compounding Analysis", "📋 Trading Schedule"])


# ==================== TAB 1: UPLOAD & BACKTEST ====================

with tab1:
    st.header("📁 Upload Your Data Files")
    st.info("Upload M5 and M15 CSV files for each pair. The backtest will use REAL historical data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇬🇧 GBP/USD")
        gbpusd_m15 = st.file_uploader("GBP/USD M15", type=['csv', 'txt'], key='gbp_m15')
        gbpusd_m5 = st.file_uploader("GBP/USD M5", type=['csv', 'txt'], key='gbp_m5')
    
    with col2:
        st.subheader("🇪🇺 EUR/USD")
        eurusd_m15 = st.file_uploader("EUR/USD M15", type=['csv', 'txt'], key='eur_m15')
        eurusd_m5 = st.file_uploader("EUR/USD M5", type=['csv', 'txt'], key='eur_m5')
    
    st.subheader("🇨🇦 USD/CAD")
    col1, col2 = st.columns(2)
    with col1:
        usdcad_m15 = st.file_uploader("USD/CAD M15", type=['csv', 'txt'], key='cad_m15')
    with col2:
        usdcad_m5 = st.file_uploader("USD/CAD M5", type=['csv', 'txt'], key='cad_m5')
    
    # Process uploaded files
    pairs_data = {}
    data_info = {'pairs': [], 'trading_days': 0, 'date_range': ''}
    
    if gbpusd_m15 and gbpusd_m5:
        try:
            content_m15 = gbpusd_m15.read().decode('utf-8')
            content_m5 = gbpusd_m5.read().decode('utf-8')
            df_m15 = parse_mt5_csv(content_m15, "GBPUSD")
            df_m5 = parse_mt5_csv(content_m5, "GBPUSD")
            if not df_m15.empty and not df_m5.empty:
                pairs_data['GBPUSD'] = {'m15': df_m15, 'm5': df_m5}
                data_info['pairs'].append('GBPUSD')
                st.success(f"✅ GBP/USD: {len(df_m15):,} M15 candles + {len(df_m5):,} M5 candles")
        except Exception as e:
            st.error(f"❌ GBP/USD error: {e}")
    
    if eurusd_m15 and eurusd_m5:
        try:
            content_m15 = eurusd_m15.read().decode('utf-8')
            content_m5 = eurusd_m5.read().decode('utf-8')
            df_m15 = parse_mt5_csv(content_m15, "EURUSD")
            df_m5 = parse_mt5_csv(content_m5, "EURUSD")
            if not df_m15.empty and not df_m5.empty:
                pairs_data['EURUSD'] = {'m15': df_m15, 'm5': df_m5}
                data_info['pairs'].append('EURUSD')
                st.success(f"✅ EUR/USD: {len(df_m15):,} M15 candles + {len(df_m5):,} M5 candles")
        except Exception as e:
            st.error(f"❌ EUR/USD error: {e}")
    
    if usdcad_m15 and usdcad_m5:
        try:
            content_m15 = usdcad_m15.read().decode('utf-8')
            content_m5 = usdcad_m5.read().decode('utf-8')
            df_m15 = parse_mt5_csv(content_m15, "USDCAD")
            df_m5 = parse_mt5_csv(content_m5, "USDCAD")
            if not df_m15.empty and not df_m5.empty:
                pairs_data['USDCAD'] = {'m15': df_m15, 'm5': df_m5}
                data_info['pairs'].append('USDCAD')
                st.success(f"✅ USD/CAD: {len(df_m15):,} M15 candles + {len(df_m5):,} M5 candles")
        except Exception as e:
            st.error(f"❌ USD/CAD error: {e}")
    
    # Update data info
    if pairs_data:
        all_dates = set()
        for pair, data in pairs_data.items():
            all_dates.update(data['m15']['date'].unique())
        all_dates = sorted(all_dates)
        
        trading_days = len([d for d in all_dates if d.strftime('%A') not in ['Saturday', 'Sunday']])
        data_info['trading_days'] = trading_days
        data_info['date_range'] = f"{min(all_dates)} to {max(all_dates)}"
        
        st.info(f"📅 Date Range: {data_info['date_range']} | Trading Days: {trading_days}")
    
    st.markdown("---")
    
    # Run Backtest Button
    if pairs_data:
        if st.button("🚀 RUN BACKTEST ON REAL DATA", type="primary", use_container_width=True):
            with st.spinner("Running backtest on your historical data..."):
                # Run backtest
                trades_df = backtest_scenario_3(pairs_data, buffer_pips)
                
                if not trades_df.empty:
                    stats = calculate_backtest_statistics(trades_df)
                    
                    st.session_state.backtest_trades = trades_df
                    st.session_state.backtest_stats = stats
                    
                    st.success(f"✅ Backtest Complete! Found {len(trades_df)} real trades.")
                else:
                    st.warning("⚠️ No trades found in the data. Check your file format and date range.")
    else:
        st.warning("👆 Please upload at least one pair's data (both M5 and M15 files).")
    
    # Display backtest results
    if st.session_state.backtest_trades is not None and not st.session_state.backtest_trades.empty:
        trades_df = st.session_state.backtest_trades
        stats = st.session_state.backtest_stats
        
        st.markdown("---")
        st.header("📊 Backtest Results (Real Data)")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", stats['total_trades'])
        with col2:
            st.metric("Wins", stats['wins'])
        with col3:
            st.metric("Losses", stats['losses'])
        with col4:
            st.metric("Win Rate", f"{stats['win_rate']}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Consecutive Wins", stats['max_consecutive_wins'])
        with col2:
            st.metric("Max Consecutive Losses", stats['max_consecutive_losses'])
        with col3:
            st.metric("Average RR", f"1:{stats['avg_rr']}")
        
        # Day-wise performance
        st.subheader("📅 Day-wise Performance")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_stats = stats['day_stats'].reindex([d for d in day_order if d in stats['day_stats'].index])
        st.dataframe(day_stats, use_container_width=True)
        
        # Pair-wise performance
        st.subheader("💱 Pair-wise Performance")
        st.dataframe(stats['pair_stats'], use_container_width=True)
        
        # Trade log
        st.subheader("📋 Complete Trade Log")
        display_cols = ['date', 'day_of_week', 'pair', 'orb_time_ist', 'direction', 'rr_ratio', 'result', 'pnl_pips']
        st.dataframe(trades_df[display_cols], use_container_width=True)


# ==================== TAB 2: COMPOUNDING ANALYSIS ====================

with tab2:
    st.header("💰 Compounding Analysis (Real Trades)")
    
    if st.session_state.backtest_trades is None or st.session_state.backtest_trades.empty:
        st.warning("⚠️ Please run the backtest first in Tab 1 to get real trade data.")
        st.info("The compounding analysis uses ACTUAL trades from your uploaded data, not simulations.")
    else:
        trades_df = st.session_state.backtest_trades
        stats = st.session_state.backtest_stats
        
        st.success(f"✅ Using {len(trades_df)} REAL trades from backtest")
        
        st.markdown("---")
        
        # Compounding method explanation
        st.subheader("📊 Three Compounding Methods")
        
        calc = RealTradesCompounding(initial_capital, risk_percent)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="neutral-card">
                <h3>🔒 Fixed Risk</h3>
                <p>Always risk same $ amount</p>
            </div>
            """, unsafe_allow_html=True)
            
            fixed_risk = calc.get_fixed_risk(initial_capital)
            st.markdown(f"**Risk:** ${fixed_risk:.2f} per trade")
            st.markdown("✅ Safest | ❌ Slowest growth")
        
        with col2:
            st.markdown("""
            <div class="milestone-card">
                <h3>📊 Milestone</h3>
                <p>Risk increases at milestones</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Risk increases as account grows:**")
            milestones = calc.get_milestone_table()[:3]
            for m in milestones:
                st.markdown(f"- {m['balance_range']} → ${m['risk']:.2f}")
            st.markdown("**✅ RECOMMENDED**")
        
        with col3:
            st.markdown("""
            <div class="compound-card">
                <h3>🚀 Full Compound</h3>
                <p>Always risk % of balance</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Risk:** {risk_percent}% of current balance")
            st.markdown("✅ Fastest growth | ⚠️ Highest risk")
        
        st.markdown("---")
        
        # Apply compounding button
        if st.button("📈 APPLY COMPOUNDING TO REAL TRADES", type="primary", use_container_width=True):
            with st.spinner("Applying compounding methods to your real trades..."):
                results = calc.compare_all_methods(trades_df)
                st.session_state.compounding_results = results
            st.success("✅ Compounding analysis complete!")
        
        # Display compounding results
        if st.session_state.compounding_results:
            results = st.session_state.compounding_results
            
            st.markdown("---")
            st.header("📊 Compounding Results (Real Trades)")
            
            # Summary cards
            col1, col2, col3 = st.columns(3)
            
            methods_info = [
                ("fixed", "🔒 Fixed Risk", col1),
                ("milestone", "📊 Milestone", col2),
                ("full_compound", "🚀 Full Compound", col3)
            ]
            
            for method, name, col in methods_info:
                if method in results:
                    r = results[method]
                    with col:
                        st.markdown(f"#### {name}")
                        
                        st.metric(
                            "Final Balance", 
                            f"${r['final_balance']:.2f}",
                            f"${r['total_profit']:+.2f}"
                        )
                        st.metric("Return", f"{r['total_return_pct']:.1f}%")
                        st.metric("Max Drawdown", f"{r['max_drawdown_pct']:.1f}%")
                        st.metric("Growth", f"{r['final_balance']/initial_capital:.1f}x")
            
            # Comparison table
            st.markdown("---")
            st.subheader("📋 Detailed Comparison")
            
            comparison_data = []
            for method in ["fixed", "milestone", "full_compound"]:
                if method in results:
                    r = results[method]
                    comparison_data.append({
                        "Method": method.replace("_", " ").title(),
                        "Initial": f"${r['initial_capital']:.2f}",
                        "Final": f"${r['final_balance']:.2f}",
                        "Profit": f"${r['total_profit']:.2f}",
                        "Return %": f"{r['total_return_pct']:.1f}%",
                        "Max DD %": f"{r['max_drawdown_pct']:.1f}%",
                        "Growth": f"{r['final_balance']/initial_capital:.1f}x",
                        "Profit Factor": f"{r['profit_factor']:.2f}"
                    })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Equity curves
            st.subheader("📈 Equity Curves (Real Performance)")
            
            fig = go.Figure()
            colors_map = {
                "fixed": "#4facfe", 
                "milestone": "#667eea", 
                "full_compound": "#f5576c"
            }
            names_map = {
                "fixed": "🔒 Fixed Risk",
                "milestone": "📊 Milestone",
                "full_compound": "🚀 Full Compound"
            }
            
            for method, r in results.items():
                fig.add_trace(go.Scatter(
                    y=r['equity_curve'],
                    mode='lines',
                    name=names_map[method],
                    line=dict(color=colors_map[method], width=2)
                ))
            
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", 
                         annotation_text="Starting Capital")
            
            fig.update_layout(
                title="Account Growth with Each Compounding Method",
                xaxis_title="Trade Number",
                yaxis_title="Account Balance ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade-by-trade details
            st.subheader("📋 Trade-by-Trade Details")
            
            method_choice = st.selectbox(
                "Select method to view details:",
                ["milestone", "fixed", "full_compound"],
                format_func=lambda x: names_map[x]
            )
            
            if method_choice in results:
                trade_details = results[method_choice]['trade_details']
                details_df = pd.DataFrame(trade_details)
                
                # Display key columns
                display_cols = ['trade_num', 'date', 'day', 'pair', 'time_ist', 'rr', 
                               'result', 'risk_used', 'pnl', 'balance_after']
                
                st.dataframe(details_df[display_cols], use_container_width=True)
            
            # Recommendation
            st.markdown("---")
            st.subheader("🎯 Recommendation")
            
            # Find best method
            best_method = max(results.items(), key=lambda x: x[1]['total_return_pct'])
            safest_method = min(results.items(), key=lambda x: x[1]['max_drawdown_pct'])
            
            st.markdown(f"""
            Based on your **REAL** backtest data ({len(trades_df)} trades):
            
            | Criteria | Best Method | Result |
            |----------|-------------|--------|
            | **Highest Return** | {names_map[best_method[0]]} | {best_method[1]['total_return_pct']:.1f}% |
            | **Lowest Drawdown** | {names_map[safest_method[0]]} | {safest_method[1]['max_drawdown_pct']:.1f}% |
            | **Recommended** | 📊 Milestone | Balanced risk/reward |
            """)


# ==================== TAB 3: TRADING SCHEDULE ====================

with tab3:
    st.header("📋 Scenario 3: Trading Schedule")
    st.markdown("**Only trades with 60%+ historical win rate**")
    
    st.markdown("---")
    
    # Display schedule
    schedule_data = []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        setups = SCENARIO_3_SETUPS.get(day, [])
        if setups:
            for setup in setups:
                schedule_data.append({
                    "Day": day,
                    "Pair": setup['pair'],
                    "Time (IST)": setup['time_ist'],
                    "Time (GMT)": setup['time_gmt'],
                    "ORB": setup['entry_mode'],
                    "RR": f"1:{setup['rr']}",
                    "Expected WR": f"{setup['expected_wr']}%"
                })
        else:
            schedule_data.append({
                "Day": day,
                "Pair": "No trades",
                "Time (IST)": "-",
                "Time (GMT)": "-",
                "ORB": "-",
                "RR": "-",
                "Expected WR": "-"
            })
    
    st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)
    
    st.markdown("---")
    
    # Weekly summary
    st.subheader("📊 Weekly Summary")
    
    total_weekly_trades = sum(len(setups) for setups in SCENARIO_3_SETUPS.values())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trades per Week", total_weekly_trades)
    
    with col2:
        all_setups = [s for setups in SCENARIO_3_SETUPS.values() for s in setups]
        avg_rr = sum(s['rr'] for s in all_setups) / len(all_setups) if all_setups else 0
        st.metric("Average RR", f"1:{avg_rr:.1f}")
    
    with col3:
        avg_wr = sum(s['expected_wr'] for s in all_setups) / len(all_setups) if all_setups else 0
        st.metric("Average Expected WR", f"{avg_wr:.1f}%")
    
    st.markdown("---")
    
    # Milestone table
    st.subheader("💰 Milestone Compounding Reference")
    
    calc = RealTradesCompounding(initial_capital, risk_percent)
    milestones = calc.get_milestone_table()
    
    milestone_display = []
    for m in milestones:
        risk = m['risk']
        milestone_display.append({
            "Balance Range": m['balance_range'],
            "Risk/Trade": f"${risk:.2f}",
            "Win 1:2": f"+${risk*2:.2f}",
            "Win 1:2.5": f"+${risk*2.5:.2f}",
            "Win 1:3": f"+${risk*3:.2f}",
            "Loss": f"-${risk:.2f}"
        })
    
    st.dataframe(pd.DataFrame(milestone_display), use_container_width=True)
    
    st.markdown("---")
    
    # Trading rules
    st.subheader("📜 Trading Rules")
    
    st.markdown("""
    1. **ORB Candle:** Mark HIGH and LOW of the 15-min candle at specified GMT time
    2. **Wait for Breakout:** Candle must CLOSE above HIGH (Long) or below LOW (Short)
    3. **Entry:** At the CLOSE of the breakout candle
    4. **Stop Loss:** Opposite side of ORB range + 2 pip buffer
    5. **Take Profit:** Risk × RR Ratio (varies by setup)
    6. **One Trade Rule:** Only take FIRST breakout (no re-entries)
    7. **Daily Limit:** Maximum 1 trade per pair per day
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
    ORB Backtester + Compounding Calculator | Uses REAL historical data only | Past performance ≠ future results
</div>
""", unsafe_allow_html=True)