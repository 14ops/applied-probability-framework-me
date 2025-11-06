"""
Wait for market open, then BUY and SELL a small quantity to validate trading.

Usage:
  python -m stock_ai.utils.execute_buy_sell_at_open --symbol AAPL --qty 1
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from loguru import logger

from stock_ai.api.alpaca_interface import AlpacaInterface


def wait_until_market_open(api: AlpacaInterface, poll_seconds: int = 15) -> None:
    while True:
        clk = api.get_clock()
        if clk.get('is_open', False):
            logger.info("Market is open. Proceeding with test trades.")
            return
        nxt = clk.get('next_open')
        logger.info(f"Market closed. Next open: {nxt}. Sleeping {poll_seconds}s...")
        time.sleep(poll_seconds)


def place_market_and_wait(api: AlpacaInterface, symbol: str, side: str, qty: int, timeout: int = 120):
    oid = api.submit_order(symbol, qty, side=side, order_type='market', time_in_force='day')
    if not oid:
        raise RuntimeError(f"Failed to submit {side} order")
    info = api.wait_for_fill(oid, timeout_seconds=timeout, poll_interval=2.0)
    if not info or info.get('status') != 'filled':
        raise RuntimeError(f"{side.capitalize()} order not filled in time: {info}")
    fill_price = info.get('filled_avg_price')
    logger.info(f"{side.upper()} filled: order={oid} price={fill_price} qty={info.get('filled_qty')}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='AAPL')
    parser.add_argument('--qty', type=int, default=1)
    args = parser.parse_args()

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger.add(str(log_dir / 'test_trade_at_open_{time:YYYY-MM-DD}.log'))

    api = AlpacaInterface()
    symbol = args.symbol.upper()
    qty = max(1, int(args.qty))

    # Wait for open
    wait_until_market_open(api)

    # BUY
    buy_info = place_market_and_wait(api, symbol, 'buy', qty)

    # Small pause to avoid immediate partial execution edge cases
    time.sleep(3)

    # SELL
    sell_info = place_market_and_wait(api, symbol, 'sell', qty)

    logger.success("Completed buy then sell at market open.")


if __name__ == '__main__':
    main()




