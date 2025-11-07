"""User-friendly Streamlit control center for Stock AI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

from stock_ai.api import BrokerFactory, BrokerNotAvailableError
from stock_ai.api.base_broker import BrokerInterface

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yaml"
LOG_DIR = PROJECT_ROOT / "logs"


def load_yaml(path: Path) -> Dict:
    try:
        with open(path, "r") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def save_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_trade_history(limit_days: Optional[int] = None) -> pd.DataFrame:
    if not LOG_DIR.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for file in sorted(LOG_DIR.glob("trade_log_*.csv")):
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    trades = pd.concat(frames, ignore_index=True)
    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
        trades.sort_values("timestamp", inplace=True)
        if limit_days is not None:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=limit_days)
            trades = trades[trades["timestamp"] >= cutoff]
    return trades


def broker_selection_sidebar() -> Dict:
    st.sidebar.header("1️⃣ Broker Setup")

    brokers = list(BrokerFactory.list_brokers())
    if not brokers:
        st.sidebar.error("No brokers registered. Install alpaca-trade-api or ib-insync.")
        st.stop()

    options = {entry.display_name: entry for entry in brokers}
    display_name = st.sidebar.selectbox("Choose a broker", list(options.keys()))
    selected = options[display_name]

    secrets = load_yaml(SECRETS_PATH)
    broker_config = secrets.setdefault(selected.name, {})

    with st.sidebar.expander("Credentials", expanded=True):
        if selected.name == "alpaca":
            broker_config["key_id"] = st.text_input("API Key", broker_config.get("key_id", ""))
            broker_config["secret_key"] = st.text_input("Secret Key", broker_config.get("secret_key", ""), type="password")
            broker_config["base_url"] = st.text_input(
                "Base URL",
                broker_config.get("base_url", "https://paper-api.alpaca.markets"),
            )
        elif selected.name == "interactive_brokers":
            broker_config["host"] = st.text_input("Host", broker_config.get("host", "127.0.0.1"))
            broker_config["port"] = st.number_input(
                "Port",
                value=int(broker_config.get("port", 7497)),
                step=1,
            )
            broker_config["client_id"] = st.number_input(
                "Client ID",
                value=int(broker_config.get("client_id", 1)),
                step=1,
            )
            broker_config["account"] = st.text_input("Account (optional)", broker_config.get("account", ""))
        else:
            st.write("This broker does not expose configuration fields yet.")

    if st.sidebar.button("Save Credentials"):
        save_yaml(SECRETS_PATH, secrets)
        st.sidebar.success("Credentials saved")

    return {"name": selected.name, "display_name": display_name, "config": broker_config}


def load_settings_ui(selected_broker: str) -> Dict:
    st.sidebar.header("2️⃣ Strategy & Risk")
    settings = load_yaml(SETTINGS_PATH)

    tickers = st.sidebar.text_input(
        "Tickers (comma separated)",
        ",".join(settings.get("tickers", ["AAPL"]))
    )

    strategy_cfg = settings.setdefault("strategy", {})
    strategies = ["ma_crossover", "rsi_filter", "bollinger_reversion", "ensemble_vote"]
    current_strategy = strategy_cfg.get("name", strategies[0])
    if current_strategy not in strategies:
        current_strategy = strategies[0]
    strategy_cfg["name"] = st.sidebar.selectbox(
        "Strategy",
        strategies,
        index=strategies.index(current_strategy),
    )
    strategy_cfg["params"] = strategy_cfg.get("params", {})

    live_cfg = settings.setdefault("live", {})
    live_cfg["check_interval"] = st.sidebar.number_input(
        "Check interval (seconds)",
        value=int(live_cfg.get("check_interval", 300)),
        step=30,
    )
    live_cfg["broker"] = selected_broker

    risk_cfg = settings.setdefault("risk", {})
    risk_cfg["max_drawdown_pct"] = st.sidebar.number_input(
        "Max drawdown %",
        value=float(risk_cfg.get("max_drawdown_pct", 5.0)),
        step=0.5,
    )
    risk_cfg["max_consecutive_losses"] = st.sidebar.number_input(
        "Max consecutive losses",
        value=int(risk_cfg.get("max_consecutive_losses", 5)),
        step=1,
    )

    if st.sidebar.button("Save Settings"):
        tickers_clean = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        settings["tickers"] = tickers_clean or ["AAPL"]
        live_cfg["broker"] = selected_broker
        save_yaml(SETTINGS_PATH, settings)
        st.sidebar.success("Settings saved")

    settings["tickers"] = [t.strip().upper() for t in settings.get("tickers", []) if t]
    return settings


def connect_broker(broker_info: Dict) -> Optional[BrokerInterface]:
    try:
        extras = {k: v for k, v in broker_info.get("config", {}).items() if v not in ("", None)}
        return BrokerFactory.create(
            broker_info["name"],
            config_path=str(SECRETS_PATH),
            **extras,
        )
    except BrokerNotAvailableError as exc:
        st.error(f"Unable to connect to {broker_info['display_name']}: {exc}")
        return None


def render_account_tab(broker_info: Dict) -> None:
    st.subheader("Broker Overview")
    if st.button("Connect & Refresh", key="refresh_account"):
        st.session_state.pop("account_cache", None)
        st.session_state.pop("positions_cache", None)
        st.session_state.pop("orders_cache", None)

    if "account_cache" not in st.session_state:
        with st.spinner("Connecting to broker..."):
            broker = connect_broker(broker_info)
            if not broker:
                return
            account = broker.get_account()
            positions = broker.get_positions()
            orders = list(broker.list_orders(status="open"))
            st.session_state["account_cache"] = account
            st.session_state["positions_cache"] = positions
            st.session_state["orders_cache"] = orders
    else:
        account = st.session_state["account_cache"]
        positions = st.session_state["positions_cache"]
        orders = st.session_state["orders_cache"]

    if not account:
        st.info("No account data available.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Cash", f"${account.get('cash', 0):,.2f}")
    col2.metric("Buying Power", f"${account.get('buying_power', 0):,.2f}")
    col3.metric("Equity", f"${account.get('equity', 0):,.2f}")

    st.markdown("### Open Positions")
    if positions:
        st.dataframe(pd.DataFrame(positions))
    else:
        st.info("No open positions")

    st.markdown("### Open Orders")
    if orders:
        st.dataframe(pd.DataFrame(orders))
    else:
        st.info("No open orders")


def render_trades_tab() -> None:
    st.subheader("Trade History")
    trades = load_trade_history()
    if trades.empty:
        st.info("No trade logs found yet. Trades appear after the live runner executes orders.")
        return

    st.markdown("### Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", len(trades))
    if "qty" in trades.columns:
        col2.metric("Total Volume", int(trades["qty"].abs().sum()))
    else:
        col2.metric("Total Volume", "-")
    if "symbol" in trades.columns:
        col3.metric("Symbols", trades["symbol"].nunique())
    else:
        col3.metric("Symbols", "-")

    symbols = sorted(set(trades.get("symbol", [])))
    if symbols:
        selected = st.multiselect("Filter symbols", symbols, default=symbols)
        trades = trades[trades["symbol"].isin(selected)]

    st.dataframe(trades)

    if "symbol" in trades.columns:
        aggregations = {"qty": "sum"}
        if "price" in trades.columns:
            aggregations["price"] = "mean"
        summary = trades.groupby("symbol").agg(aggregations)
        summary.rename(columns={"qty": "Total Shares", "price": "Avg Price"}, inplace=True)
        st.markdown("### Summary by Symbol")
        st.dataframe(summary)


def render_quick_trade_tab(broker_info: Dict) -> None:
    st.subheader("Quick Trade Ticket")
    with st.form("trade_form"):
        symbol = st.text_input("Symbol", value="AAPL").upper()
        qty = st.number_input("Quantity", min_value=1, value=1)
        side = st.selectbox("Side", ["buy", "sell"])
        order_type = st.selectbox("Order Type", ["market", "limit", "stop"])
        limit_price = st.number_input("Limit Price", value=0.0)
        stop_price = st.number_input("Stop Price", value=0.0)
        submitted = st.form_submit_button("Submit Order")

    if submitted:
        with st.spinner("Sending order..."):
            broker = connect_broker(broker_info)
            if not broker:
                return
            kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "order_type": order_type,
            }
            if order_type == "limit" and limit_price > 0:
                kwargs["limit_price"] = limit_price
            if order_type == "stop" and stop_price > 0:
                kwargs["stop_price"] = stop_price
            result = broker.submit_order(**kwargs)
            if result.order_id:
                st.success(f"Order submitted! ID: {result.order_id} (status: {result.status})")
            else:
                st.error(f"Order failed: {result.status}")


def render_settings_tab(settings: Dict) -> None:
    st.subheader("Current Configuration Snapshot")
    st.json(settings)

    st.markdown("### Raw YAML (advanced)")
    raw_text = yaml.safe_dump(settings, sort_keys=False)
    updated_text = st.text_area("settings.yaml", raw_text, height=240)
    if st.button("Save YAML", key="save_raw_settings"):
        try:
            parsed = yaml.safe_load(updated_text) or {}
            save_yaml(SETTINGS_PATH, parsed)
            st.success("settings.yaml updated")
        except yaml.YAMLError as exc:
            st.error(f"Invalid YAML: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Stock AI Control Center",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 Stock AI Control Center")
    st.caption("Manage strategies, brokers, and live trading without touching code.")

    broker_info = broker_selection_sidebar()
    settings = load_settings_ui(broker_info["name"])
    settings.setdefault("live", {})["broker"] = broker_info["name"]

    tabs = st.tabs(["Dashboard", "Trades", "Quick Trade", "Settings"])
    with tabs[0]:
        render_account_tab(broker_info)
    with tabs[1]:
        render_trades_tab()
    with tabs[2]:
        render_quick_trade_tab(broker_info)
    with tabs[3]:
        render_settings_tab(settings)

    st.sidebar.header("3️⃣ Launch")
    st.sidebar.markdown(
        """
        * Run `python run_dashboard.py` to open this panel again
        * Run `python run_batch.py` for bulk backtests
        * Run `python -m stock_ai.live_trading.live_runner` to start automated trading
        """
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
