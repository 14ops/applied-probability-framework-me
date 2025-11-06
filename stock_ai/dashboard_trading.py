"""
Paper Trading Dashboard
Real-time monitoring GUI for paper trading using Streamlit
"""

# Suppress warnings BEFORE importing anything that might generate them
import warnings
import logging
import os

# Set environment variables FIRST, before any imports
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

# Configure root logger to suppress warnings
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s',
    force=True  # Override any existing configuration
)

# Suppress all UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')

# Custom filter to suppress ScriptRunContext messages
class ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        msg = str(record.getMessage())
        return 'ScriptRunContext' not in msg and 'missing ScriptRunContext' not in msg

# Apply filter to all Streamlit loggers (will be set up after streamlit import)
def setup_streamlit_logging():
    """Configure Streamlit loggers to suppress warnings."""
    loggers_to_suppress = [
        'streamlit',
        'streamlit.runtime',
        'streamlit.runtime.caching',
        'streamlit.runtime.caching.cache_data_api',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.scriptrunner.script_runner',
        'streamlit.runtime.scriptrunner_utils.script_run_context'
    ]
    filter_instance = ScriptRunContextFilter()
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        logger.addFilter(filter_instance)
        logger.setLevel(logging.ERROR)
    
    # Also suppress "No runtime found" cache warnings specifically
    class CacheWarningFilter(logging.Filter):
        def filter(self, record):
            msg = str(record.getMessage())
            return 'No runtime found' not in msg and 'MemoryCacheStorageManager' not in msg
    
    cache_logger = logging.getLogger('streamlit.runtime.caching.cache_data_api')
    cache_logger.addFilter(CacheWarningFilter())
    cache_logger.setLevel(logging.ERROR)

# Now import streamlit and other packages
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yaml
from pathlib import Path
import sys
import subprocess
import platform
import signal
from typing import Optional

# Setup Streamlit logging after import
setup_streamlit_logging()

# Additional suppression for warnings that appear during import
import warnings
warnings.simplefilter("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_ai.api.alpaca_interface import AlpacaInterface

# Initialize page configuration only when in Streamlit context
def init_page_config():
    """Initialize Streamlit page configuration."""
    try:
        st.set_page_config(
            page_title="Paper Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass  # Already configured or not in Streamlit context

# Custom CSS for better styling
CUSTOM_CSS = """
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 5px;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff3333;
    }
</style>
"""

def apply_custom_css():
    """Apply custom CSS styling."""
    try:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    except Exception:
        pass  # Not in Streamlit context


# -------------------------
# Helpers & IO utilities
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STATE_DIR = PROJECT_ROOT / "run_state"
PID_FILE = RUN_STATE_DIR / "live_runner.pid"


def _ensure_dirs():
    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict:
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return {}


def write_yaml(path: Path, data: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Failed to write {path}: {e}")
        return False


def load_secrets() -> dict:
    return read_yaml(PROJECT_ROOT / "config" / "secrets.yaml")


def save_secrets(alpaca_key: str, alpaca_secret: str, base_url: str) -> bool:
    secrets = read_yaml(PROJECT_ROOT / "config" / "secrets.yaml")
    secrets.setdefault('alpaca', {})
    secrets['alpaca']['key_id'] = (alpaca_key or '').strip()
    secrets['alpaca']['secret_key'] = (alpaca_secret or '').strip()
    secrets['alpaca']['base_url'] = (base_url or 'https://paper-api.alpaca.markets').strip()
    return write_yaml(PROJECT_ROOT / "config" / "secrets.yaml", secrets)


def load_settings() -> dict:
    return read_yaml(PROJECT_ROOT / "config" / "settings.yaml")


def save_settings(updated: dict) -> bool:
    return write_yaml(PROJECT_ROOT / "config" / "settings.yaml", updated)


def get_python_executable() -> str:
    # Prefer local venv if present
    venv_py = PROJECT_ROOT / ".venv"
    if platform.system() == 'Windows':
        candidate = venv_py / "Scripts" / "python.exe"
    else:
        candidate = venv_py / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if platform.system() == 'Windows':
            # Use tasklist to check
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in out.stdout
        else:
            # Signal 0 check
            os.kill(pid, 0)  # type: ignore
            return True
    except Exception:
        return False


def get_saved_pid() -> int:
    try:
        if PID_FILE.exists():
            return int(PID_FILE.read_text().strip())
    except Exception:
        return -1
    return -1


def start_live_trading() -> int:
    _ensure_dirs()
    # Remove kill switch if present
    settings = load_settings()
    kill_file = settings.get('risk', {}).get('kill_switch_file', 'stop.txt')
    try:
        (PROJECT_ROOT / kill_file).unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    python_exe = get_python_executable()
    runner = PROJECT_ROOT / "stock_ai" / "live_trading" / "live_runner.py"
    creationflags = 0
    startupinfo = None
    if platform.system() == 'Windows':
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        proc = subprocess.Popen(
            [python_exe, str(runner), "--auto-confirm"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        PID_FILE.write_text(str(proc.pid))
        return proc.pid
    except Exception as e:
        st.error(f"Failed to start trading bot: {e}")
        return -1


def stop_live_trading(grace_seconds: int = 10) -> bool:
    settings = load_settings()
    kill_file = settings.get('risk', {}).get('kill_switch_file', 'stop.txt')
    # Create kill switch to ask runner to stop
    try:
        (PROJECT_ROOT / kill_file).write_text("stop")
    except Exception:
        pass
    pid = get_saved_pid()
    # Wait a bit for graceful shutdown
    for _ in range(grace_seconds):
        if pid > 0 and not is_process_running(pid):
            break
        time.sleep(1)
    # Force kill if still running
    if pid > 0 and is_process_running(pid):
        try:
            if platform.system() == 'Windows':
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGTERM)  # type: ignore
        except Exception:
            pass
    try:
        PID_FILE.unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    return True


def is_bot_running() -> bool:
    pid = get_saved_pid()
    if pid > 0 and is_process_running(pid):
        return True
    # Fallback: check logs freshness (existing behavior)
    try:
        log_dir = PROJECT_ROOT / "logs"
        log_files = list(log_dir.glob("live_trading_*.log"))
        if log_files:
            latest = max(log_files, key=lambda p: p.stat().st_mtime)
            return (time.time() - latest.stat().st_mtime) < 600
    except Exception:
        pass
    return False


def dependency_installed(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


def install_requirements():
    try:
        python_exe = get_python_executable()
        req = PROJECT_ROOT / "requirements.txt"
        subprocess.check_call([python_exe, "-m", "pip", "install", "-r", str(req)])
        return True
    except Exception as e:
        st.error(f"Failed to install requirements: {e}")
        return False


def compute_limit_price(api: AlpacaInterface, symbol: str, side: str, pad_pct: float = 0.01) -> Optional[float]:
    """Compute a reasonable limit price using quote or last bar as fallback."""
    quote = None
    try:
        quote = api.get_quote(symbol)
    except Exception:
        quote = None
    price = None
    if quote and (quote.get('ask_price') or quote.get('bid_price')):
        if side == 'buy':
            base = quote.get('ask_price') or quote.get('bid_price')
            price = base * (1 + abs(pad_pct))
        else:
            base = quote.get('bid_price') or quote.get('ask_price')
            price = base * (1 - abs(pad_pct))
    else:
        # Fallback to latest minute bar
        bars = api.get_bars([symbol], '1Min', limit=1)
        if bars and symbol in bars and len(bars[symbol]) > 0:
            last_close = bars[symbol][-1]['close']
            if side == 'buy':
                price = last_close * (1 + abs(pad_pct))
            else:
                price = last_close * (1 - abs(pad_pct))
    if price and price > 0:
        return round(price, 2)
    return None


def place_manual_order(symbol: str, qty: int, side: str) -> str:
    """Place a manual/test order, using market if open, limit otherwise."""
    api = AlpacaInterface()
    clk = api.get_clock()
    is_open = bool(clk.get('is_open', False))
    symbol = symbol.upper().strip()
    qty = max(1, int(qty))
    if is_open:
        oid = api.submit_order(symbol, qty, side=side, order_type='market', time_in_force='day')
        if not oid:
            return "Failed to submit market order"
        info = api.wait_for_fill(oid, timeout_seconds=90, poll_interval=2.0)
        return f"Order {oid} status: {info.get('status') if info else 'unknown'}"
    # Market closed -> try extended-hours limit
    limit_price = compute_limit_price(api, symbol, side, pad_pct=0.01)
    if not limit_price:
        return "Could not compute a limit price (no quote/bar)"
    oid = api.submit_order(
        symbol, qty, side=side, order_type='limit', time_in_force='day', limit_price=limit_price, extended_hours=True
    )
    if not oid:
        return "Failed to submit extended-hours limit order"
    info = api.wait_for_fill(oid, timeout_seconds=10, poll_interval=2.0)
    if info and info.get('status') == 'filled':
        return f"Order {oid} filled at {info.get('filled_avg_price')}"
    return f"Order {oid} submitted (market closed) — will fill when executable."

def get_account_data():
    """Get current account data from Alpaca."""
    try:
        api = AlpacaInterface()
        account = api.get_account()
        positions = api.get_positions()
        clock = api.api.get_clock()
        return account, positions, clock
    except Exception as e:
        # Only show error if we're in a Streamlit context
        try:
            st.error(f"Error connecting to Alpaca: {e}")
        except:
            pass  # Not in Streamlit context yet
        return None, [], None

def get_recent_orders(limit=50):
    """Get recent orders from Alpaca API (real-time)."""
    try:
        api = AlpacaInterface()
        orders = api.get_orders(status='all', limit=limit)
        # Convert to DataFrame format
        order_data = []
        for order in orders:
            # Get timestamp (prefer filled_at if available, otherwise submitted_at)
            timestamp_str = order.get('filled_at') or order.get('submitted_at') or order.get('created_at', '')
            try:
                timestamp = pd.Timestamp(timestamp_str) if timestamp_str else pd.Timestamp.now()
            except:
                timestamp = pd.Timestamp.now()
            
            order_data.append({
                'timestamp': timestamp,
                'symbol': order.get('symbol', ''),
                'action': order.get('side', '').upper(),
                'qty': order.get('qty', 0),
                'order_type': order.get('type', ''),
                'status': order.get('status', ''),
                'filled_qty': order.get('filled_qty', 0),
                'filled_avg_price': order.get('filled_avg_price', 0) or 0,
                'order_id': order.get('id', order.get('order_id', ''))
            })
        if order_data:
            df = pd.DataFrame(order_data)
            # Sort by timestamp, most recent first
            df = df.sort_values('timestamp', ascending=False)
            return df
        return pd.DataFrame()
    except Exception as e:
        try:
            st.warning(f"Could not load orders: {e}")
        except:
            pass
        return pd.DataFrame()

def get_pending_orders():
    """Get pending/open orders from Alpaca."""
    try:
        api = AlpacaInterface()
        orders = api.get_orders(status='open', limit=20)
        return orders
    except Exception as e:
        return []

# Cache function only when Streamlit context exists
try:
    get_account_data = st.cache_data(ttl=2)(get_account_data)  # Cache for 2 seconds for real-time feel
    get_recent_orders = st.cache_data(ttl=3)(get_recent_orders)  # Cache for 3 seconds
except:
    pass  # Not in Streamlit context, use uncached version


def load_equity_history():
    """Load equity history from CSV file."""
    try:
        # Find the most recent equity history file
        log_dir = Path("logs")
        equity_files = list(log_dir.glob("equity_history_*.csv"))
        if not equity_files:
            return pd.DataFrame()
        
        latest_file = max(equity_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.warning(f"Could not load equity history: {e}")
        return pd.DataFrame()


def load_trades_log():
    """Load trades log from CSV file."""
    try:
        trades_file = Path("results/trades_log.csv")
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load trades log: {e}")
        return pd.DataFrame()


def load_live_trading_log():
    """Load recent entries from live trading log."""
    try:
        log_dir = Path("logs")
        log_files = list(log_dir.glob("live_trading_*.log"))
        if not log_files:
            return []
        
        latest_file = max(log_files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, 'r') as f:
            lines = f.readlines()
            # Get last 50 lines
            return lines[-50:]
    except Exception as e:
        st.warning(f"Could not load trading log: {e}")
        return []


def load_config():
    """Load trading configuration."""
    try:
        with open("config/settings.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.warning(f"Could not load config: {e}")
        return {}


def create_equity_chart(df):
    """Create equity curve chart."""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add drawdown shading if max_equity column exists
    if 'max_equity' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['max_equity'],
            mode='lines',
            name='Peak Equity',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.5
        ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    
    return fig


def main():
    """Main dashboard function."""
    
    # Initialize Streamlit components
    init_page_config()
    apply_custom_css()
    
    # Header
    st.title("📈 Paper Trading Dashboard")
    st.markdown("---")
    
    # Quick checks and helper banner
    st.sidebar.header("🚀 Quick Actions")
    if not dependency_installed('alpaca_trade_api'):
        st.sidebar.error("alpaca-trade-api not installed")
        if st.sidebar.button("Install requirements"):
            with st.spinner("Installing packages... this may take a minute"):
                if install_requirements():
                    st.success("Installed. Please rerun.")
                    st.rerun()
    
    # Start/Stop controls
    bot_active = is_bot_running()
    if bot_active:
        st.sidebar.success("🤖 Trading Bot: ACTIVE")
        if st.sidebar.button("⏹️ Stop Bot"):
            with st.spinner("Stopping bot..."):
                stop_live_trading()
                st.success("Bot stopped")
                st.rerun()
    else:
        st.sidebar.warning("🤖 Trading Bot: INACTIVE")
        if st.sidebar.button("▶️ Start Bot"):
            with st.spinner("Starting bot..."):
                pid = start_live_trading()
                if pid > 0:
                    st.success(f"Bot started (PID {pid})")
                st.rerun()
    
    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
        time.sleep(refresh_interval)
        st.rerun()
    else:
        if st.sidebar.button("🔄 Refresh"):
            st.rerun()
    
    # Load data
    account, positions, clock = get_account_data()
    equity_df = load_equity_history()
    trades_df = load_trades_log()
    config = load_config()
    
    # Get REAL-TIME orders from Alpaca API
    realtime_orders_df = get_recent_orders(limit=50)
    pending_orders = get_pending_orders()
    
    if account is None:
        st.error("⚠️ Could not connect to Alpaca API. Please check your credentials.")
        return
    
    # Sidebar - Status and Config
    st.sidebar.header("⚙️ Status")
    
    # Market status
    if clock:
        try:
            market_status = "🟢 OPEN" if clock.is_open else "🔴 CLOSED"
            st.sidebar.markdown(f"**Market:** {market_status}")
            if hasattr(clock, 'next_open') and clock.next_open:
                try:
                    next_open = pd.Timestamp(clock.next_open).strftime("%Y-%m-%d %H:%M")
                    st.sidebar.caption(f"Next open: {next_open}")
                except:
                    pass
        except:
            st.sidebar.markdown("**Market:** Status unknown")
    
    # Trading config display
    st.sidebar.header("📋 Configuration")
    if config:
        strategy = config.get('strategy', {}).get('name', 'N/A')
        tickers = config.get('tickers', [])
        st.sidebar.caption(f"**Strategy:** {strategy}")
        st.sidebar.caption(f"**Symbols:** {', '.join(tickers)}")
        st.sidebar.caption(f"**Risk Limit:** {config.get('risk', {}).get('max_drawdown_pct', 5)}% drawdown")
        # Quick risk actions
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            if st.button("Create Kill Switch"):
                ks = config.get('risk', {}).get('kill_switch_file', 'stop.txt')
                (PROJECT_ROOT / ks).write_text("stop")
                st.sidebar.info("Kill switch created")
        with col_b:
            if st.button("Remove Kill Switch"):
                ks = config.get('risk', {}).get('kill_switch_file', 'stop.txt')
                try:
                    (PROJECT_ROOT / ks).unlink()
                    st.sidebar.success("Removed")
                except Exception:
                    st.sidebar.warning("Not found")
        # Close positions quick action
        if st.sidebar.button("🛑 Close All Positions"):
            try:
                api = AlpacaInterface()
                ok = api.close_all_positions()
                if ok:
                    st.sidebar.success("All positions closed")
                else:
                    st.sidebar.warning("Close positions failed")
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Portfolio metrics
    with col1:
        portfolio_value = account.get('portfolio_value', 0)
        st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    
    with col2:
        cash = account.get('cash', 0)
        st.metric("Cash", f"${cash:,.2f}")
    
    with col3:
        buying_power = account.get('buying_power', 0)
        st.metric("Buying Power", f"${buying_power:,.2f}")
    
    with col4:
        equity = account.get('equity', 0)
        st.metric("Equity", f"${equity:,.2f}")
    
    st.markdown("---")
    
    # Equity curve and positions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Equity Curve")
        if not equity_df.empty:
            fig = create_equity_chart(equity_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity history data yet. Equity curve will appear after first trading cycle.")
    
    with col2:
        st.subheader("💼 Open Positions")
        if positions:
            pos_data = []
            for pos in positions:
                pnl_color = "positive" if pos['unrealized_pl'] >= 0 else "negative"
                pnl_sign = "+" if pos['unrealized_pl'] >= 0 else ""
                pos_data.append({
                    'Symbol': pos['symbol'],
                    'Qty': pos['qty'],
                    'Price': f"${pos['current_price']:.2f}",
                    'P&L': f"{pnl_sign}${pos['unrealized_pl']:.2f}",
                    'P&L %': f"{pos['unrealized_plpc']*100:+.2f}%"
                })
            pos_df = pd.DataFrame(pos_data)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")
    
    st.markdown("---")
    
    # Pending Orders Alert
    if pending_orders:
        st.warning(f"⚠️ {len(pending_orders)} Pending Order(s) - Waiting for execution")
        with st.expander("View Pending Orders", expanded=False):
            for order in pending_orders[:10]:
                status_icon = "⏳" if order.get('status') == 'open' else "🔄"
                st.text(f"{status_icon} {order.get('symbol')} - {order.get('side', '').upper()} {order.get('qty')} @ {order.get('type', '')}")
    
    st.markdown("---")
    
    # Recent trades and risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Live Trades (Real-Time from Alpaca)")
        if not realtime_orders_df.empty:
            # Show only filled/completed orders
            filled_orders = realtime_orders_df[
                (realtime_orders_df['status'] == 'filled') | 
                (realtime_orders_df['status'] == 'partially_filled')
            ].head(20)
            
            if not filled_orders.empty:
                # Format display
                display_orders = filled_orders.copy()
                display_orders['Time'] = display_orders['timestamp'].apply(
                    lambda x: pd.Timestamp(x).strftime("%H:%M:%S") if pd.notna(x) else ""
                )
                display_orders['Date'] = display_orders['timestamp'].apply(
                    lambda x: pd.Timestamp(x).strftime("%m/%d") if pd.notna(x) else ""
                )
                
                # Create styled display
                for idx, order in filled_orders.iterrows():
                    action_emoji = "🟢 BUY" if order['action'] == 'BUY' else "🔴 SELL"
                    status_emoji = "✅" if order['status'] == 'filled' else "🟡"
                    price_str = f"${order['filled_avg_price']:.2f}" if order['filled_avg_price'] > 0 else "Pending"
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_a:
                        st.markdown(f"**{action_emoji}**")
                    with col_b:
                        st.markdown(f"**{order['symbol']}** {order['filled_qty']}/{order['qty']} @ {price_str}")
                    with col_c:
                        time_str = pd.Timestamp(order['timestamp']).strftime("%H:%M:%S") if pd.notna(order['timestamp']) else ""
                        st.caption(f"{status_emoji} {time_str}")
                
                # Also show as table
                with st.expander("📊 View All Orders Table", expanded=False):
                    table_data = filled_orders[['timestamp', 'symbol', 'action', 'filled_qty', 'filled_avg_price', 'status']].copy()
                    table_data['timestamp'] = table_data['timestamp'].apply(
                        lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
                    )
                    table_data['filled_avg_price'] = table_data['filled_avg_price'].apply(
                        lambda x: f"${x:.2f}" if x > 0 else "N/A"
                    )
                    st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.info("No filled orders yet (orders are pending)")
        else:
            # Fallback to CSV log
            if not trades_df.empty:
                st.info("Showing trades from log file (Alpaca API orders not available)")
                recent_trades = trades_df.head(10)[['timestamp', 'symbol', 'action', 'qty', 'price', 'order_id']]
                recent_trades['price'] = recent_trades['price'].apply(lambda x: f"${x:.2f}")
                recent_trades['timestamp'] = recent_trades['timestamp'].apply(lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                st.dataframe(recent_trades, use_container_width=True, hide_index=True)
            else:
                st.info("No trades executed yet")
    
    with col2:
        st.subheader("⚠️ Risk Metrics")
        
        # Calculate risk metrics
        if not equity_df.empty and 'drawdown_pct' in equity_df.columns:
            current_drawdown = equity_df['drawdown_pct'].iloc[-1]
            max_drawdown = equity_df['drawdown_pct'].max()
            
            col_a, col_b = st.columns(2)
            with col_a:
                drawdown_color = "🟢" if current_drawdown < 2 else "🟡" if current_drawdown < 4 else "🔴"
                st.metric("Current Drawdown", f"{drawdown_color} {current_drawdown:.2f}%")
            with col_b:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        # Show account status
        if account.get('trading_blocked', False):
            st.error("⚠️ Trading is blocked")
        if account.get('account_blocked', False):
            st.error("🚫 Account is blocked")
        if account.get('pattern_day_trader', False):
            st.warning("⚠️ Pattern Day Trader")
    
    st.markdown("---")
    
    # Live trading log
    st.subheader("📋 Live Trading Log")
    log_lines = load_live_trading_log()
    if log_lines:
        # Show last 20 lines
        log_text = "".join(log_lines[-20:])
        st.text_area("Recent log entries", log_text, height=200, key="log_area")
    else:
        st.info("No log entries available")
    
    st.markdown("---")
    
    # Settings & Broker Setup
    tabs = st.tabs(["⚙️ Settings", "🔐 Broker Setup"])
    
    with tabs[0]:
        st.subheader("Configuration")
        settings = load_settings()
        current_tickers = settings.get('tickers', [])
        tickers_str = ", ".join(current_tickers)
        col_a, col_b = st.columns(2)
        with col_a:
            tickers_input = st.text_input("Symbols (comma-separated)", value=tickers_str)
            check_interval = st.number_input("Check interval (seconds)", min_value=30, max_value=3600, value=int(settings.get('live', {}).get('check_interval', 300)), step=10)
            allow_after_hours = st.checkbox("Allow after-hours trading (testing)", value=bool(settings.get('live', {}).get('allow_after_hours', False)))
        with col_b:
            max_dd = st.number_input("Max drawdown % (kill-switch)", min_value=1.0, max_value=50.0, value=float(settings.get('risk', {}).get('max_drawdown_pct', 5.0)), step=0.5)
            max_losses = st.number_input("Max consecutive losses", min_value=1, max_value=20, value=int(settings.get('risk', {}).get('max_consecutive_losses', 5)))
            ks_file = st.text_input("Kill switch file", value=str(settings.get('risk', {}).get('kill_switch_file', 'stop.txt')))
        if st.button("Save Settings"):
            updated = settings.copy()
            # Update tickers
            updated['tickers'] = [s.strip().upper() for s in tickers_input.split(',') if s.strip()]
            # Live
            updated.setdefault('live', {})
            updated['live']['check_interval'] = int(check_interval)
            updated['live']['allow_after_hours'] = bool(allow_after_hours)
            # Risk
            updated.setdefault('risk', {})
            updated['risk']['max_drawdown_pct'] = float(max_dd)
            updated['risk']['max_consecutive_losses'] = int(max_losses)
            updated['risk']['kill_switch_file'] = ks_file.strip() or 'stop.txt'
            if save_settings(updated):
                st.success("Settings saved")
            else:
                st.error("Failed to save settings")
    
    with tabs[1]:
        st.subheader("Alpaca Credentials")
        secrets = load_secrets()
        alp = secrets.get('alpaca', {})
        key_id = st.text_input("API Key ID", value=str(alp.get('key_id', '')))
        secret_key = st.text_input("API Secret Key", value=str(alp.get('secret_key', '')), type="password")
        base_url = st.text_input("Base URL", value=str(alp.get('base_url', 'https://paper-api.alpaca.markets')))
        col1x, col2x = st.columns(2)
        with col1x:
            if st.button("Save Credentials"):
                if save_secrets(key_id, secret_key, base_url):
                    st.success("Saved credentials")
                else:
                    st.error("Failed to save credentials")
        with col2x:
            if st.button("Test Connection"):
                try:
                    api = AlpacaInterface()
                    acct = api.get_account()
                    st.success(f"Connected. Portfolio Value: ${acct.get('portfolio_value', 0):,.2f}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        st.caption("Keys are stored locally in config/secrets.yaml and never uploaded.")
        st.markdown("---")
        st.subheader("🧪 Quick Test / Manual Trade")
        mcol1, mcol2, mcol3 = st.columns([2,1,2])
        with mcol1:
            man_symbol = st.text_input("Symbol", value="AAPL")
        with mcol2:
            man_qty = st.number_input("Quantity", min_value=1, max_value=1000, value=1, step=1)
        with mcol3:
            st.caption("Uses market order when open; limit with extended-hours when closed.")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            if st.button("Buy 1x Test"):
                msg = place_manual_order(man_symbol, int(man_qty), side='buy')
                st.info(msg)
        with tcol2:
            if st.button("Sell 1x Test"):
                msg = place_manual_order(man_symbol, int(man_qty), side='sell')
                st.info(msg)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("💡 Tip: Enable auto-refresh for real-time monitoring")


if __name__ == "__main__":
    main()

