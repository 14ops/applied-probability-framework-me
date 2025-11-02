"""
Streamlit Dashboard

Interactive dashboard for monitoring and analyzing trading strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    st.error("matplotlib not installed. Install with: pip install matplotlib")
    st.stop()

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Stock AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def load_summary_data():
    """Load backtest summary data."""
    summary_path = Path("results/summary.csv")
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def load_backtest_result(ticker: str):
    """Load individual backtest result."""
    backtest_path = Path(f"results/backtest_{ticker}.csv")
    if backtest_path.exists():
        df = pd.read_csv(backtest_path, index_col=0, parse_dates=True)
        return df
    return None


def load_equity_history():
    """Load equity history from live trading."""
    logs_dir = Path("logs")
    equity_files = list(logs_dir.glob("equity_history_*.csv"))
    
    if not equity_files:
        return None
    
    # Load most recent file
    latest_file = max(equity_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, parse_dates=True)
    return df


def load_trade_log():
    """Load trade log from live trading."""
    logs_dir = Path("logs")
    trade_files = list(logs_dir.glob("trade_log_*.csv"))
    
    if not trade_files:
        return None
    
    # Load most recent file
    latest_file = max(trade_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, parse_dates=True)
    return df


def plot_equity_over_time(df: pd.DataFrame):
    """Plot equity over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'timestamp' in df.columns:
        ax.plot(df['timestamp'], df['equity'], linewidth=2, color='blue')
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
    else:
        ax.plot(df.index, df['equity'], linewidth=2, color='blue')
        ax.set_xlabel('Date')
    
    ax.set_ylabel('Equity ($)')
    ax.set_title('Equity Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig


def plot_drawdown_over_time(df: pd.DataFrame):
    """Plot drawdown over time."""
    if 'drawdown' not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    drawdown_pct = df['drawdown'] * 100
    ax.fill_between(df.index, drawdown_pct, 0, alpha=0.3, color='red')
    ax.plot(df.index, drawdown_pct, linewidth=2, color='darkred')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">📈 Stock AI Trading Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Backtest Analysis", "Live Trading", "Strategy Performance", "Settings"]
        )
        
        st.markdown("---")
        st.header("Quick Stats")
        
        # Load summary data
        summary_df = load_summary_data()
        
        if summary_df is not None:
            avg_sharpe = summary_df['sharpe_ratio'].mean()
            avg_return = summary_df['total_return_pct'].mean()
            avg_drawdown = summary_df['max_drawdown_pct'].mean()
            
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.3f}")
            st.metric("Avg Return", f"{avg_return:.2f}%")
            st.metric("Avg Max DD", f"{avg_drawdown:.2f}%")
        else:
            st.info("No backtest data available")
    
    # Main content based on selected page
    if page == "Backtest Analysis":
        show_backtest_analysis(summary_df)
    elif page == "Live Trading":
        show_live_trading()
    elif page == "Strategy Performance":
        show_strategy_performance(summary_df)
    elif page == "Settings":
        show_settings()


def show_backtest_analysis(summary_df):
    """Backtest analysis page."""
    st.header("Backtest Analysis")
    
    if summary_df is None:
        st.warning("No backtest summary data found. Run batch_runner.py first.")
        return
    
    # Summary table
    st.subheader("Performance Summary")
    st.dataframe(summary_df, use_container_width=True)
    
    # Best and worst performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Best Performers")
        if 'total_return_pct' in summary_df.columns:
            top_3 = summary_df.nlargest(3, 'total_return_pct')[['ticker', 'total_return_pct', 'sharpe_ratio']]
            st.dataframe(top_3, use_container_width=True)
    
    with col2:
        st.subheader("📉 Worst Performers")
        if 'total_return_pct' in summary_df.columns:
            bottom_3 = summary_df.nsmallest(3, 'total_return_pct')[['ticker', 'total_return_pct', 'sharpe_ratio']]
            st.dataframe(bottom_3, use_container_width=True)
    
    # Detailed analysis
    st.markdown("---")
    st.subheader("Detailed Analysis")
    
    selected_ticker = st.selectbox("Select Ticker", summary_df['ticker'].tolist())
    
    bt_df = load_backtest_result(selected_ticker)
    if bt_df is not None:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", summary_df[summary_df['ticker'] == selected_ticker]['num_trades'].values[0])
        with col2:
            st.metric("Win Rate", f"{summary_df[summary_df['ticker'] == selected_ticker]['win_rate_pct'].values[0]:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{summary_df[summary_df['ticker'] == selected_ticker]['sharpe_ratio'].values[0]:.3f}")
        with col4:
            st.metric("Max Drawdown", f"{summary_df[summary_df['ticker'] == selected_ticker]['max_drawdown_pct'].values[0]:.2f}%")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(plot_equity_over_time(bt_df))
        
        with col2:
            drawdown_fig = plot_drawdown_over_time(bt_df)
            if drawdown_fig:
                st.pyplot(drawdown_fig)
    
    else:
        st.warning(f"No detailed data found for {selected_ticker}")


def show_live_trading():
    """Live trading monitoring page."""
    st.header("Live Trading Monitor")
    
    # Load live trading data
    equity_df = load_equity_history()
    trade_df = load_trade_log()
    
    if equity_df is None:
        st.info("No live trading data available. Start the live_runner to begin tracking.")
        return
    
    # Current portfolio metrics
    if equity_df is not None and len(equity_df) > 0:
        latest = equity_df.iloc[-1]
        initial = equity_df.iloc[0] if len(equity_df) > 1 else latest
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Equity", f"${latest['equity']:,.2f}")
        with col2:
            pnl = latest['equity'] - initial['equity']
            st.metric("Total P&L", f"${pnl:,.2f}")
        with col3:
            pnl_pct = ((latest['equity'] - initial['equity']) / initial['equity']) * 100
            st.metric("Return %", f"{pnl_pct:.2f}%")
        with col4:
            st.metric("Cash", f"${latest['cash']:,.2f}")
        
        # Plot equity history
        st.subheader("Equity History")
        st.pyplot(plot_equity_over_time(equity_df))
    
    # Recent trades
    if trade_df is not None and len(trade_df) > 0:
        st.subheader("Recent Trades")
        st.dataframe(trade_df.tail(20), use_container_width=True)
    else:
        st.info("No trades logged yet")


def show_strategy_performance(summary_df):
    """Strategy performance comparison page."""
    st.header("Strategy Performance Comparison")
    
    if summary_df is None:
        st.warning("No backtest data available")
        return
    
    # Plot comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return vs Sharpe")
        if all(col in summary_df.columns for col in ['total_return_pct', 'sharpe_ratio']):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(summary_df['sharpe_ratio'], summary_df['total_return_pct'], s=100)
            
            for _, row in summary_df.iterrows():
                ax.annotate(row['ticker'], (row['sharpe_ratio'], row['total_return_pct']))
            
            ax.set_xlabel('Sharpe Ratio')
            ax.set_ylabel('Total Return (%)')
            ax.set_title('Risk-Adjusted Return Comparison')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.subheader("Drawdown Distribution")
        if 'max_drawdown_pct' in summary_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(summary_df['ticker'], summary_df['max_drawdown_pct'], color='coral')
            ax.set_xlabel('Max Drawdown (%)')
            ax.set_title('Maximum Drawdown by Ticker')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)


def show_settings():
    """Settings configuration page."""
    st.header("Settings")
    
    st.subheader("Configuration Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Settings (config/settings.yaml)**")
        settings_path = Path("config/settings.yaml")
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                st.code(f.read(), language='yaml')
        else:
            st.warning("Settings file not found")
    
    with col2:
        st.markdown("**Secrets Template**")
        secrets_template_path = Path("config/secrets.yaml.template")
        if secrets_template_path.exists():
            with open(secrets_template_path, 'r') as f:
                st.code(f.read(), language='yaml')
        else:
            st.warning("Secrets template not found")
    
    st.info("⚠️ Never commit secrets.yaml to version control!")


if __name__ == "__main__":
    main()

