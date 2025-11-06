"""
Report Generator

Generates comprehensive PDF and HTML reports from backtest results,
combining metrics, charts, and optimization summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from loguru import logger

try:
    from matplotlib.backends.backend_pdf import PdfPages
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PDF backend not available, reports will be HTML only")


def load_backtest_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all backtest CSV files from results directory."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    # Find all backtest CSV files
    backtest_files = list(results_path.glob("backtest_*.csv"))
    
    if not backtest_files:
        logger.warning(f"No backtest files found in {results_dir}")
        return pd.DataFrame()
    
    # Load and combine
    dfs = []
    for file in backtest_files:
        try:
            df = pd.read_csv(file)
            # Extract ticker from filename
            ticker = file.stem.replace("backtest_", "")
            df['ticker'] = ticker
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def load_optimization_summary(results_dir: str = "results") -> pd.DataFrame:
    """Load optimization summary CSV."""
    summary_path = Path(results_dir) / "optimization_summary.csv"
    
    if not summary_path.exists():
        logger.warning(f"Optimization summary not found: {summary_path}")
        return pd.DataFrame()
    
    try:
        return pd.read_csv(summary_path)
    except Exception as e:
        logger.error(f"Error loading optimization summary: {e}")
        return pd.DataFrame()


def load_summary_csv(results_dir: str = "results") -> pd.DataFrame:
    """Load summary CSV from batch backtests."""
    summary_path = Path(results_dir) / "summary.csv"
    
    if not summary_path.exists():
        logger.warning(f"Summary CSV not found: {summary_path}")
        return pd.DataFrame()
    
    try:
        return pd.read_csv(summary_path)
    except Exception as e:
        logger.error(f"Error loading summary CSV: {e}")
        return pd.DataFrame()


def create_equity_curve_plot(backtest_df: pd.DataFrame, output_path: Optional[str] = None):
    """Create equity curve plot from backtest data."""
    if backtest_df.empty or 'equity' not in backtest_df.columns:
        logger.warning("Cannot create equity curve: missing data or equity column")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot equity curves for each ticker
    if 'ticker' in backtest_df.columns:
        for ticker in backtest_df['ticker'].unique():
            ticker_data = backtest_df[backtest_df['ticker'] == ticker]
            if 'date' in ticker_data.columns:
                ax.plot(ticker_data['date'], ticker_data['equity'], label=ticker, alpha=0.7)
            else:
                ax.plot(ticker_data.index, ticker_data['equity'], label=ticker, alpha=0.7)
    else:
        if 'date' in backtest_df.columns:
            ax.plot(backtest_df['date'], backtest_df['equity'], alpha=0.7)
        else:
            ax.plot(backtest_df.index, backtest_df['equity'], alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.set_title('Equity Curve Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curve to {output_path}")
    
    return fig


def create_performance_comparison(summary_df: pd.DataFrame, output_path: Optional[str] = None):
    """Create performance comparison charts."""
    if summary_df.empty:
        logger.warning("Cannot create performance comparison: empty summary")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total Return
    if 'total_return_pct' in summary_df.columns and 'ticker' in summary_df.columns:
        axes[0, 0].bar(summary_df['ticker'], summary_df['total_return_pct'])
        axes[0, 0].set_title('Total Return by Ticker')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Sharpe Ratio
    if 'sharpe_ratio' in summary_df.columns and 'ticker' in summary_df.columns:
        axes[0, 1].bar(summary_df['ticker'], summary_df['sharpe_ratio'])
        axes[0, 1].set_title('Sharpe Ratio by Ticker')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Max Drawdown
    if 'max_drawdown_pct' in summary_df.columns and 'ticker' in summary_df.columns:
        axes[1, 0].bar(summary_df['ticker'], summary_df['max_drawdown_pct'])
        axes[1, 0].set_title('Max Drawdown by Ticker')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Win Rate
    if 'win_rate_pct' in summary_df.columns and 'ticker' in summary_df.columns:
        axes[1, 1].bar(summary_df['ticker'], summary_df['win_rate_pct'])
        axes[1, 1].set_title('Win Rate by Ticker')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance comparison to {output_path}")
    
    return fig


def create_optimization_summary_plot(opt_df: pd.DataFrame, output_path: Optional[str] = None):
    """Create visualization of optimization results."""
    if opt_df.empty:
        logger.warning("Cannot create optimization plot: empty data")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top 10 by Sharpe Ratio
    if 'sharpe_ratio' in opt_df.columns:
        top_sharpe = opt_df.nlargest(10, 'sharpe_ratio')
        axes[0, 0].barh(range(len(top_sharpe)), top_sharpe['sharpe_ratio'])
        axes[0, 0].set_yticks(range(len(top_sharpe)))
        axes[0, 0].set_yticklabels([f"{row.get('strategy', '')} {row.get('symbol', '')}" 
                                    for _, row in top_sharpe.iterrows()], fontsize=8)
        axes[0, 0].set_title('Top 10 by Sharpe Ratio')
        axes[0, 0].set_xlabel('Sharpe Ratio')
    
    # Return vs Drawdown scatter
    if 'total_return_pct' in opt_df.columns and 'max_drawdown_pct' in opt_df.columns:
        axes[0, 1].scatter(opt_df['max_drawdown_pct'], opt_df['total_return_pct'], alpha=0.5)
        axes[0, 1].set_xlabel('Max Drawdown (%)')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].set_title('Return vs Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Return distribution
    if 'total_return_pct' in opt_df.columns:
        axes[1, 0].hist(opt_df['total_return_pct'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Total Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Strategy comparison
    if 'strategy' in opt_df.columns and 'sharpe_ratio' in opt_df.columns:
        strategy_sharpe = opt_df.groupby('strategy')['sharpe_ratio'].mean()
        axes[1, 1].bar(strategy_sharpe.index, strategy_sharpe.values)
        axes[1, 1].set_title('Average Sharpe Ratio by Strategy')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved optimization summary plot to {output_path}")
    
    return fig


def generate_html_report(
    summary_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    output_path: str = "reports/report.html"
):
    """Generate HTML report with metrics and charts."""
    html_path = Path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock AI Trading Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>Stock AI Trading Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary Statistics</h2>
"""
    
    if not summary_df.empty:
        html_content += summary_df.to_html(index=False, classes='table')
    else:
        html_content += "<p>No summary data available.</p>"
    
    html_content += """
    <h2>Optimization Results</h2>
"""
    
    if not opt_df.empty:
        html_content += opt_df.head(20).to_html(index=False, classes='table')
    else:
        html_content += "<p>No optimization data available.</p>"
    
    html_content += """
    <h2>Charts</h2>
    <p>Charts saved separately as PNG files in the reports directory.</p>
    
    <footer>
        <p><em>Generated by Stock AI Trading Framework</em></p>
    </footer>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {html_path}")


def generate_pdf_report(
    summary_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    output_path: str = "reports/report.pdf"
):
    """Generate PDF report with metrics and charts."""
    if not PDF_AVAILABLE:
        logger.warning("PDF generation not available, generating HTML instead")
        generate_html_report(summary_df, opt_df, output_path.replace('.pdf', '.html'))
        return
    
    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(str(pdf_path)) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.5, f'Stock AI Trading Report\n\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=20, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Summary table
        if not summary_df.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax.set_title('Backtest Summary', fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Charts
        if not backtest_df.empty:
            equity_fig = create_equity_curve_plot(backtest_df)
            if equity_fig:
                pdf.savefig(equity_fig, bbox_inches='tight')
                plt.close(equity_fig)
        
        if not summary_df.empty:
            perf_fig = create_performance_comparison(summary_df)
            if perf_fig:
                pdf.savefig(perf_fig, bbox_inches='tight')
                plt.close(perf_fig)
        
        if not opt_df.empty:
            opt_fig = create_optimization_summary_plot(opt_df)
            if opt_fig:
                pdf.savefig(opt_fig, bbox_inches='tight')
                plt.close(opt_fig)
    
    logger.info(f"PDF report generated: {pdf_path}")


def generate_report(
    results_dir: str = "results",
    reports_dir: str = "reports",
    format: str = "both"  # "pdf", "html", or "both"
):
    """
    Generate comprehensive trading report.
    
    Args:
        results_dir: Directory containing backtest and optimization results
        reports_dir: Directory to save report outputs
        format: Output format ("pdf", "html", or "both")
    """
    logger.info("Generating trading report...")
    
    # Load data
    summary_df = load_summary_csv(results_dir)
    opt_df = load_optimization_summary(results_dir)
    backtest_df = load_backtest_results(results_dir)
    
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Generate charts
    if not backtest_df.empty:
        create_equity_curve_plot(backtest_df, str(reports_path / "equity_curves.png"))
    
    if not summary_df.empty:
        create_performance_comparison(summary_df, str(reports_path / "performance_comparison.png"))
    
    if not opt_df.empty:
        create_optimization_summary_plot(opt_df, str(reports_path / "optimization_summary.png"))
    
    # Generate reports
    if format in ["pdf", "both"]:
        generate_pdf_report(summary_df, opt_df, backtest_df, str(reports_path / "report.pdf"))
    
    if format in ["html", "both"]:
        generate_html_report(summary_df, opt_df, str(reports_path / "report.html"))
    
    logger.info("Report generation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading reports")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--reports-dir", default="reports", help="Reports output directory")
    parser.add_argument("--format", choices=["pdf", "html", "both"], default="both",
                       help="Report format")
    args = parser.parse_args()
    
    generate_report(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        format=args.format
    )

