# 📈 Paper Trading Dashboard

A real-time monitoring GUI for your paper trading bot built with Streamlit.

## 🚀 Quick Start

Run the dashboard:
```bash
python run_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## ✨ Features

### 📊 Real-Time Monitoring
- **Portfolio Metrics**: Live portfolio value, cash, buying power, and equity
- **Market Status**: Clearly indicates when the market is open or closed
- **Equity Curve**: Interactive chart for portfolio performance over time
- **Open Positions**: Current holdings with real-time P&L tracking
- **Trade Blotter**: Filterable view of fills/orders from any connected broker
- **Risk Metrics**: Current and max drawdown, account alerts, and broker warnings
- **Live Logs**: Stream of trading activity for quick diagnostics

### 🔌 Broker Flexibility
- **Broker Registry**: Choose Alpaca or upload CSV exports from other brokers
- **Manual/CSV Mode**: Point the dashboard at any trade history file to review fills
- **Credential Management**: Update broker-specific settings directly in the UI

### 🎛️ Controls
- **Auto-refresh**: Automatic updates on a configurable interval
- **Manual refresh**: One-click refresh whenever auto mode is disabled
- **Sidebar**: Connection status, broker controls, and trading bot start/stop buttons

## 📋 Dashboard Sections

### 1. Overview Tab
- Portfolio cards for value, cash, buying power, and equity
- Market status with next open/close times
- Equity curve chart
- Live positions table and pending order summary
- Optional quick trade ticket for brokers that support order placement

### 2. Trade Blotter Tab
- Searchable, filterable list of recent broker fills/orders
- Downloadable CSV for audit trails
- Secondary section displaying locally saved trade logs
- CSV uploader for the Manual/CSV broker mode

### 3. Insights Tab
- Drawdown metrics and health checks
- Highlighted broker alerts (trading blocked, PDT status, etc.)
- Live trading log viewer

### 4. Settings Tab
- Strategy configuration (tickers, refresh cadence, risk limits)
- Broker-specific credential forms stored in `config/dashboard.yaml`
- Direct editor/tester for Alpaca API keys stored in `config/secrets.yaml`

## ⚙️ Configuration

Key files used by the dashboard:

- `config/settings.yaml` – Core trading strategy configuration shared with the live runner
- `config/dashboard.yaml` – Dashboard preferences, active broker, and broker credentials
- `config/secrets.yaml` – Alpaca API keys (when using the Alpaca broker)
- `logs/equity_history_*.csv` – Equity history data
- `results/trades_log.csv` – Locally generated trade log (used as a fallback)
- `logs/live_trading_*.log` – Real-time trading logs

## 🔧 Troubleshooting

### Dashboard won't start
- Ensure Streamlit and Plotly are installed: `pip install streamlit plotly`
- Check that paper trading is running (logs should be generating)

### No data showing
- Wait for the first trading cycle to complete
- Check that `logs/` and `results/` directories exist
- Verify paper trading bot is actively running

### Connection errors
- Use the **Retry connection** button in the sidebar
- Verify broker credentials in the Settings tab
- For Alpaca, confirm API keys in `config/secrets.yaml`

## 📱 Access

The dashboard runs locally on:
- **URL**: http://localhost:8501
- **Port**: 8501 (default)

You can access it from any browser on your machine.

## 🛑 Stopping the Dashboard

Press `Ctrl+C` in the terminal where the dashboard is running.

---

**Note**: The dashboard updates automatically when auto-refresh is enabled. For best performance, keep it open while paper trading is active.

