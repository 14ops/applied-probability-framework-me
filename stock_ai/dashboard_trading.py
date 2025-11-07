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
from typing import Any, Dict, List, Optional, Tuple

# Setup Streamlit logging after import
setup_streamlit_logging()

# Additional suppression for warnings that appear during import
import warnings
warnings.simplefilter("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_ai.api import BrokerError, REGISTRY

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
DASHBOARD_CONFIG_PATH = PROJECT_ROOT / "config" / "dashboard.yaml"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "active_broker": "alpaca",
    "brokers": {
        "alpaca": {
            "config_path": "config/secrets.yaml",
        },
        "manual_csv": {
            "trades_csv": "",
            "account_label": "My Brokerage Account",
            "cash": 0.0,
            "equity": 0.0,
            "buying_power": 0.0,
        },
    },
}

NUMERIC_BROKER_FIELDS = {"cash", "equity", "buying_power"}


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


def _merge_settings(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_settings(result[key], value)
        else:
            result[key] = value
    return result


def load_dashboard_settings() -> Dict[str, Any]:
    stored = read_yaml(DASHBOARD_CONFIG_PATH)
    return _merge_settings(DEFAULT_SETTINGS, stored)


def save_dashboard_settings(updated: Dict[str, Any]) -> bool:
    existing = read_yaml(DASHBOARD_CONFIG_PATH)
    merged = _merge_settings(DEFAULT_SETTINGS, existing)
    merged = _merge_settings(merged, updated)
    return write_yaml(DASHBOARD_CONFIG_PATH, merged)


def load_strategy_settings() -> Dict[str, Any]:
    return read_yaml(PROJECT_ROOT / "config" / "settings.yaml")


def save_strategy_settings(updated: Dict[str, Any]) -> bool:
    return write_yaml(PROJECT_ROOT / "config" / "settings.yaml", updated)


BROKER_CACHE: Dict[str, Any] = {}


def get_broker_client(settings: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Initialise the broker client defined in settings."""

    broker_id = settings.get("active_broker", DEFAULT_SETTINGS["active_broker"])
    definition = REGISTRY.get(broker_id)
    if definition is None:
        return None, f"Broker '{broker_id}' is not registered"

    broker_settings = settings.get("brokers", {}).get(broker_id, {})
    cache_key = (broker_id, tuple(sorted(broker_settings.items())))
    cached = BROKER_CACHE.get(cache_key)
    if cached is not None:
        return cached, None

    try:
        client = REGISTRY.create(broker_id, broker_settings)
        client.connect()
        BROKER_CACHE[cache_key] = client
        return client, None
    except (BrokerError, ValueError) as exc:
        BROKER_CACHE.pop(cache_key, None)
        return None, str(exc)


def safe_call(client: Any, method: str, *args, default=None, **kwargs):
    if client is None:
        return default
    func = getattr(client, method, None)
    if not callable(func):
        return default
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        try:
            st.warning(f"Broker error calling {method}: {exc}")
        except Exception:  # Not in Streamlit context
            pass
        return default


def format_currency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${number:,.2f}"


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
    settings = load_strategy_settings()
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
    settings = load_strategy_settings()
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


def place_manual_order(client: Any, symbol: str, qty: int, side: str) -> str:
    """Place an order through the connected broker."""

    if client is None:
        return "No broker connected"

    submit = getattr(client, "submit_order", None)
    if not callable(submit):
        return f"{getattr(client, 'display_name', 'Broker')} does not support order placement from the dashboard"

    symbol = symbol.upper().strip()
    qty = max(1, int(qty))

    try:
        order_id = submit(symbol=symbol, qty=qty, side=side, order_type="market")
    except Exception as exc:  # pylint: disable=broad-except
        return f"Order failed: {exc}"

    if not order_id:
        return "Order submitted but broker did not return an ID"
    return f"Order {order_id} submitted successfully"

def get_account_data(client: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Fetch account information from the active broker."""

    account = safe_call(client, "get_account", default=None)
    positions = safe_call(client, "list_positions", default=[]) or []
    clock = safe_call(client, "get_clock", default=None)
    return account, positions, clock


def _normalise_orders(raw_orders: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_orders:
        return pd.DataFrame()
    order_data = []
    for order in raw_orders:
        timestamp_str = (
            order.get("filled_at")
            or order.get("submitted_at")
            or order.get("timestamp")
            or order.get("created_at")
            or ""
        )
        try:
            timestamp = pd.to_datetime(timestamp_str) if timestamp_str else pd.Timestamp.now()
        except Exception:  # pylint: disable=broad-except
            timestamp = pd.Timestamp.now()
        order_data.append(
            {
                "timestamp": timestamp,
                "symbol": order.get("symbol", order.get("Symbol", "")),
                "action": (order.get("side") or order.get("Side", "")).upper(),
                "qty": float(order.get("qty", order.get("Quantity", 0)) or 0),
                "order_type": order.get("type", order.get("Order Type", "")),
                "status": order.get("status", order.get("Status", "")),
                "filled_qty": float(order.get("filled_qty", order.get("Filled Qty", 0)) or 0),
                "filled_avg_price": float(order.get("filled_avg_price", order.get("Price", 0)) or 0),
                "order_id": order.get("id", order.get("order_id", order.get("Order ID", ""))),
            }
        )
    df = pd.DataFrame(order_data)
    return df.sort_values("timestamp", ascending=False)


def get_recent_orders(client: Any, limit: int = 50) -> pd.DataFrame:
    """Get recent orders from the selected broker."""

    if client is None:
        return pd.DataFrame()

    orders = safe_call(client, "list_orders", status="all", limit=limit, default=None)
    if orders is None:
        orders = safe_call(client, "list_orders", default=[])
    if not orders:
        return pd.DataFrame()
    return _normalise_orders(orders)


def get_pending_orders(client: Any) -> List[Dict[str, Any]]:
    orders = safe_call(client, "list_orders", status="open", limit=20, default=None)
    if orders is None:
        orders = safe_call(client, "list_orders", default=[])
    return orders or []


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
        return load_strategy_settings()
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
    """Render the Streamlit dashboard."""

    init_page_config()
    apply_custom_css()
    _ensure_dirs()

    st.title("📊 Trading Control Center")
    st.caption("Monitor live trades, review history, and manage broker credentials without writing code.")
    st.markdown("---")

    dashboard_settings = load_dashboard_settings()
    strategy_settings = load_strategy_settings()
    broker_id = dashboard_settings.get("active_broker", DEFAULT_SETTINGS["active_broker"])
    broker_client, broker_error = get_broker_client(dashboard_settings)
    broker_definitions = REGISTRY.list_definitions()

    definition_by_name = {definition.display_name: definition for definition in broker_definitions}
    selected_definition = next((d for d in broker_definitions if d.broker_id == broker_id), broker_definitions[0] if broker_definitions else None)

    with st.sidebar:
        st.header("🔌 Broker connection")
        if broker_definitions:
            display_names = list(definition_by_name.keys())
            default_index = display_names.index(selected_definition.display_name) if selected_definition else 0
            choice = st.selectbox("Active broker", display_names, index=default_index)
            chosen_definition = definition_by_name[choice]
            if chosen_definition.broker_id != broker_id:
                save_dashboard_settings({"active_broker": chosen_definition.broker_id})
                BROKER_CACHE.clear()
                st.success(f"Switched to {chosen_definition.display_name}. Reloading...")
                st.rerun()
            st.caption(chosen_definition.description)
            selected_definition = chosen_definition
        else:
            st.warning("No broker implementations registered. Add one in stock_ai/api/.")
            selected_definition = None

        if st.button("🔄 Retry connection"):
            BROKER_CACHE.clear()
            st.rerun()

        if broker_error:
            st.error(f"Connection failed: {broker_error}")
        elif selected_definition:
            st.success(f"Connected to {selected_definition.display_name}")

        st.divider()
        st.header("🤖 Automation")
        if not dependency_installed('alpaca_trade_api'):
            st.warning("alpaca-trade-api not installed")
            if st.button("Install requirements"):
                with st.spinner("Installing required packages..."):
                    if install_requirements():
                        st.success("Dependencies installed. Please rerun.")
                        st.rerun()
        else:
            st.caption("alpaca-trade-api available")

        bot_active = is_bot_running()
        if bot_active:
            st.success("Trading bot running")
            if st.button("⏹️ Stop bot"):
                with st.spinner("Stopping live trading..."):
                    stop_live_trading()
                st.success("Bot stopped")
                st.rerun()
        else:
            st.info("Trading bot stopped")
            if st.button("▶️ Start bot"):
                with st.spinner("Starting live trading..."):
                    pid = start_live_trading()
                if pid > 0:
                    st.success(f"Bot started (PID {pid})")
                st.rerun()

        st.divider()
        st.header("🔁 Refresh")
        auto_refresh = st.checkbox("Auto refresh", value=True)
        refresh_interval = st.slider("Refresh every", min_value=5, max_value=60, value=10, step=5)
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        elif st.button("Refresh now"):
            st.rerun()

    account, positions, clock = (None, [], None)
    if broker_client:
        account, positions, clock = get_account_data(broker_client)

    equity_df = load_equity_history()
    trades_df = load_trades_log()
    realtime_orders_df = get_recent_orders(broker_client, limit=200)
    pending_orders = get_pending_orders(broker_client)
    pending_df = _normalise_orders(pending_orders)

    overview_tab, trades_tab, insights_tab, settings_tab = st.tabs([
        "Overview",
        "Trade Blotter",
        "Insights",
        "Settings",
    ])

    with overview_tab:
        st.subheader("Account snapshot")
        if broker_error or account is None:
            st.warning("Unable to load account information. Check your credentials in the Settings tab.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio value", format_currency(account.get("portfolio_value")))
            with col2:
                st.metric("Cash", format_currency(account.get("cash")))
            with col3:
                st.metric("Buying power", format_currency(account.get("buying_power")))
            with col4:
                st.metric("Equity", format_currency(account.get("equity")))

        clock_info = {}
        if clock:
            if isinstance(clock, dict):
                clock_info = clock
            else:
                clock_info = {
                    "is_open": getattr(clock, "is_open", None),
                    "next_open": getattr(clock, "next_open", None),
                    "next_close": getattr(clock, "next_close", None),
                }

        if clock_info:
            status = "🟢 Market open" if clock_info.get("is_open") else "🔴 Market closed"
            st.info(status)
            if clock_info.get("next_open"):
                st.caption(f"Next open: {pd.Timestamp(clock_info['next_open']).strftime('%Y-%m-%d %H:%M')}")
            if clock_info.get("next_close"):
                st.caption(f"Next close: {pd.Timestamp(clock_info['next_close']).strftime('%Y-%m-%d %H:%M')}")

        st.markdown("---")
        st.subheader("Equity curve")
        if not equity_df.empty:
            st.plotly_chart(create_equity_chart(equity_df), use_container_width=True)
        else:
            st.info("Run the strategy or import equity history to see performance.")

        st.subheader("Open positions")
        if positions:
            pos_df = pd.DataFrame(positions)
            rename_map = {
                "symbol": "Symbol",
                "qty": "Quantity",
                "market_value": "Market Value",
                "current_price": "Price",
                "unrealized_pl": "Unrealised P/L",
                "unrealized_plpc": "Unrealised %",
                "side": "Side",
            }
            pos_df = pos_df.rename(columns=rename_map)
            if "Unrealised %" in pos_df.columns:
                pos_df["Unrealised %"] = pos_df["Unrealised %"].astype(float) * 100
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions reported by the broker.")

        if not pending_df.empty:
            st.subheader("Pending orders")
            st.dataframe(pending_df[["timestamp", "symbol", "action", "qty", "status"]], use_container_width=True, hide_index=True)

        if hasattr(broker_client, "submit_order") and callable(getattr(broker_client, "submit_order")):
            with st.expander("Quick trade ticket"):
                qcol1, qcol2, qcol3 = st.columns([2, 1, 1])
                with qcol1:
                    ticket_symbol = st.text_input("Symbol", value="AAPL", key="ticket_symbol")
                with qcol2:
                    ticket_qty = st.number_input("Quantity", min_value=1, max_value=10_000, value=1, step=1, key="ticket_qty")
                with qcol3:
                    action = st.selectbox("Side", ["buy", "sell"], index=0, key="ticket_side")
                if st.button("Submit order", key="ticket_submit"):
                    message = place_manual_order(broker_client, ticket_symbol, int(ticket_qty), side=action)
                    st.info(message)

    with trades_tab:
        st.subheader("Live orders and fills")
        if realtime_orders_df.empty:
            st.info("No orders received from the broker yet.")
        else:
            available_status = sorted(realtime_orders_df['status'].dropna().unique()) if 'status' in realtime_orders_df else []
            status_filter = st.multiselect("Status filter", available_status, default=available_status)
            symbol_filter = st.text_input("Symbol filter", value="", placeholder="e.g. AAPL")
            filtered = realtime_orders_df.copy()
            if status_filter:
                filtered = filtered[filtered['status'].isin(status_filter)]
            if symbol_filter:
                filtered = filtered[filtered['symbol'].str.contains(symbol_filter.upper(), case=False, na=False)]
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            csv_data = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv_data, file_name="broker_orders.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Imported trade log")
        if trades_df.empty:
            st.info("No trade log found in results/trades_log.csv")
        else:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

        if selected_definition and selected_definition.broker_id == "manual_csv":
            st.markdown("### Upload fresh fills from your broker")
            uploaded = st.file_uploader("Drop a CSV exported from your broker", type="csv")
            if uploaded is not None:
                dest = PROJECT_ROOT / "data" / f"manual_trades_{int(time.time())}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(uploaded.getvalue())
                st.success(f"Saved upload to {dest}. Click 'Save broker settings' to use it.")
                st.session_state["uploaded_manual_trades"] = str(dest)

    with insights_tab:
        st.subheader("Risk & diagnostics")
        if not equity_df.empty and 'drawdown_pct' in equity_df.columns:
            drawdown = equity_df['drawdown_pct']
            st.metric("Current drawdown", f"{drawdown.iloc[-1]:.2f}%")
            st.metric("Max drawdown", f"{drawdown.max():.2f}%")
        else:
            st.info("Drawdown data not available yet.")

        if account:
            if account.get('trading_blocked'):
                st.error("Trading is currently blocked on your broker account.")
            if account.get('account_blocked'):
                st.error("Account is blocked. Contact your broker.")
            if account.get('pattern_day_trader'):
                st.warning("Broker flags you as a pattern day trader.")

        st.markdown("---")
        st.subheader("Live trading log")
        log_lines = load_live_trading_log()
        if log_lines:
            st.text_area("Recent log entries", "".join(log_lines[-40:]), height=220, key="live_log")
        else:
            st.info("No log entries yet. Start the bot to generate live logs.")

    with settings_tab:
        st.subheader("Strategy configuration")
        current_tickers = (strategy_settings or {}).get('tickers', [])
        tickers_str = ", ".join(current_tickers)
        col1, col2 = st.columns(2)
        with col1:
            tickers_input = st.text_input("Symbols (comma separated)", value=tickers_str)
            check_interval = st.number_input(
                "Check interval (seconds)",
                min_value=30,
                max_value=3600,
                value=int((strategy_settings or {}).get('live', {}).get('check_interval', 300)),
                step=30,
            )
        with col2:
            max_dd = st.number_input(
                "Max drawdown %",
                min_value=1.0,
                max_value=50.0,
                value=float((strategy_settings or {}).get('risk', {}).get('max_drawdown_pct', 5.0)),
                step=0.5,
            )
            max_losses = st.number_input(
                "Max consecutive losses",
                min_value=1,
                max_value=20,
                value=int((strategy_settings or {}).get('risk', {}).get('max_consecutive_losses', 5)),
            )
        ks_file = st.text_input(
            "Kill switch file",
            value=str((strategy_settings or {}).get('risk', {}).get('kill_switch_file', 'stop.txt')),
        )
        allow_after_hours = st.checkbox(
            "Allow after-hours trading (testing only)",
            value=bool((strategy_settings or {}).get('live', {}).get('allow_after_hours', False)),
        )
        if st.button("Save strategy settings"):
            updated = strategy_settings.copy() if strategy_settings else {}
            updated['tickers'] = [sym.strip().upper() for sym in tickers_input.split(',') if sym.strip()]
            updated.setdefault('live', {})
            updated['live']['check_interval'] = int(check_interval)
            updated['live']['allow_after_hours'] = bool(allow_after_hours)
            updated.setdefault('risk', {})
            updated['risk']['max_drawdown_pct'] = float(max_dd)
            updated['risk']['max_consecutive_losses'] = int(max_losses)
            updated['risk']['kill_switch_file'] = ks_file.strip() or 'stop.txt'
            if save_strategy_settings(updated):
                st.success("Strategy settings saved")
            else:
                st.error("Failed to save settings")

        st.markdown("---")
        if selected_definition:
            st.subheader(f"{selected_definition.display_name} settings")
            broker_settings = dashboard_settings.get('brokers', {}).get(selected_definition.broker_id, {})
            broker_updates: Dict[str, Any] = {}
            for field_name, help_text in {**selected_definition.required_credentials, **selected_definition.optional_credentials}.items():
                label = help_text or field_name.replace('_', ' ').title()
                current_value = broker_settings.get(field_name, "")
                key = f"{selected_definition.broker_id}_{field_name}"
                if field_name in NUMERIC_BROKER_FIELDS:
                    broker_updates[field_name] = st.number_input(label, value=float(current_value or 0.0), key=key)
                else:
                    broker_updates[field_name] = st.text_input(label, value=str(current_value or ""), key=key)

            uploaded_manual_path = st.session_state.get("uploaded_manual_trades")
            if uploaded_manual_path and selected_definition.broker_id == "manual_csv":
                st.info(f"Using uploaded trades file: {uploaded_manual_path}")
                broker_updates['trades_csv'] = uploaded_manual_path

            if st.button("Save broker settings"):
                save_payload = {
                    "brokers": {
                        selected_definition.broker_id: broker_updates
                    }
                }
                if save_dashboard_settings(save_payload):
                    st.success("Broker settings saved")
                    BROKER_CACHE.clear()
                    st.session_state.pop("uploaded_manual_trades", None)
                else:
                    st.error("Could not save broker settings")

            if selected_definition.broker_id == "alpaca":
                st.markdown("---")
                st.subheader("Alpaca API keys")
                secrets = load_secrets()
                alp = secrets.get('alpaca', {})
                key_id = st.text_input("API Key ID", value=str(alp.get('key_id', '')), key="alpaca_key")
                secret_key = st.text_input("API Secret Key", value=str(alp.get('secret_key', '')), type="password", key="alpaca_secret")
                base_url = st.text_input("Base URL", value=str(alp.get('base_url', 'https://paper-api.alpaca.markets')), key="alpaca_base")
                if st.button("Save Alpaca keys"):
                    if save_secrets(key_id, secret_key, base_url):
                        st.success("Credentials saved to config/secrets.yaml")
                    else:
                        st.error("Could not save credentials")
                if st.button("Test Alpaca connection"):
                    try:
                        from stock_ai.api.alpaca_interface import AlpacaInterface  # Local import to avoid circular

                        api = AlpacaInterface()
                        acct = api.get_account()
                        st.success(f"Connected. Portfolio value {format_currency(acct.get('portfolio_value'))}")
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Connection failed: {exc}")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("💡 Tip: keep auto refresh enabled while the bot is live for real-time monitoring.")


if __name__ == "__main__":
    main()

