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
- **Market Status**: Shows if market is open or closed
- **Equity Curve**: Interactive chart showing portfolio performance over time
- **Open Positions**: Current positions with P&L tracking
- **Recent Trades**: Last 10 trades with full details
- **Risk Metrics**: Current and max drawdown tracking
- **Live Logs**: Real-time trading log feed

### 🎛️ Controls
- **Auto-refresh**: Automatically updates every 5 seconds (configurable)
- **Manual refresh**: Click refresh button when auto-refresh is off
- **Sidebar**: Shows market status, strategy config, and risk limits

## 📋 Dashboard Sections

### 1. Portfolio Overview
- Portfolio Value, Cash, Buying Power, Equity in metric cards

### 2. Equity Curve
- Interactive Plotly chart showing portfolio value over time
- Includes peak equity line to visualize drawdowns

### 3. Open Positions
- Table showing all current positions
- Displays: Symbol, Quantity, Current Price, P&L ($), P&L (%)

### 4. Recent Trades
- Last 10 executed trades
- Shows: Timestamp, Symbol, Action (buy/sell), Quantity, Price, Order ID

### 5. Risk Metrics
- Current drawdown percentage
- Maximum drawdown reached
- Trading status indicators

### 6. Live Trading Log
- Real-time feed of trading activity
- Last 20 log entries displayed

## ⚙️ Configuration

The dashboard reads from:
- `config/settings.yaml` - Trading configuration
- `logs/equity_history_*.csv` - Equity history data
- `results/trades_log.csv` - Trade history
- `logs/live_trading_*.log` - Live trading logs

## 🔧 Troubleshooting

### Dashboard won't start
- Ensure Streamlit and Plotly are installed: `pip install streamlit plotly`
- Check that paper trading is running (logs should be generating)

### No data showing
- Wait for the first trading cycle to complete
- Check that `logs/` and `results/` directories exist
- Verify paper trading bot is actively running

### Connection errors
- Verify Alpaca credentials in `config/secrets.yaml`
- Check that API keys are valid and haven't expired

## 📱 Access

The dashboard runs locally on:
- **URL**: http://localhost:8501
- **Port**: 8501 (default)

You can access it from any browser on your machine.

## 🛑 Stopping the Dashboard

Press `Ctrl+C` in the terminal where the dashboard is running.

---

**Note**: The dashboard updates automatically when auto-refresh is enabled. For best performance, keep it open while paper trading is active.

