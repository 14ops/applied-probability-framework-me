"""
Quick test trade script to BUY then SELL a small quantity for validation.

Usage:
  python -m stock_ai.utils.test_trade --symbol AAPL --qty 1
"""

import argparse
from datetime import datetime
from time import sleep
from loguru import logger

from stock_ai.api.alpaca_interface import AlpacaInterface


def place_limit_with_padding(api: AlpacaInterface, symbol: str, side: str, pad_pct: float = 0.01):
    quote = api.get_quote(symbol)
    if not quote or not (quote.get('ask_price') or quote.get('bid_price')):
        # Fallback to last bar
        bars = api.get_bars([symbol], '1Min', limit=1)
        last_close = None
        if bars and symbol in bars and len(bars[symbol]) > 0:
            last_close = bars[symbol][-1]['close']
        if not last_close:
            raise RuntimeError("No quote/bar available for limit order")
        if side == 'buy':
            price = last_close * (1 + abs(pad_pct))
        else:
            price = last_close * (1 - abs(pad_pct))
    else:
        if side == 'buy':
            price = (quote.get('ask_price') or quote.get('bid_price')) * (1 + abs(pad_pct))
        else:
            price = (quote.get('bid_price') or quote.get('ask_price')) * (1 - abs(pad_pct))
    if not price or price <= 0:
        raise RuntimeError("Invalid price computed for limit order")
    return round(price, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='AAPL')
    parser.add_argument('--qty', type=int, default=1)
    args = parser.parse_args()

    api = AlpacaInterface()
    clock = api.get_clock()
    is_open = clock.get('is_open', False)
    logger.info(f"Market open: {is_open}")

    symbol = args.symbol.upper()
    qty = max(1, int(args.qty))

    # BUY
    if is_open:
        logger.info(f"Placing MARKET BUY {qty} {symbol}")
        buy_id = api.submit_order(symbol, qty, side='buy', order_type='market', time_in_force='day')
    else:
        price = place_limit_with_padding(api, symbol, 'buy', pad_pct=0.01)
        logger.info(f"Placing LIMIT BUY {qty} {symbol} @ {price} (extended hours)")
        buy_id = api.submit_order(symbol, qty, side='buy', order_type='limit', time_in_force='day', limit_price=price, extended_hours=True)

    if not buy_id:
        raise SystemExit("Failed to submit BUY order")

    info = api.wait_for_fill(buy_id, timeout_seconds=90 if is_open else 10)
    logger.info(f"BUY status: {info}")
    if not info or info.get('status') != 'filled':
        if not is_open:
            raise SystemExit("Buy not filled (market closed). Re-run during market hours for guaranteed fill.")
        raise SystemExit("Buy did not fill in time.")

    # SELL
    sleep(2)
    if is_open:
        logger.info(f"Placing MARKET SELL {qty} {symbol}")
        sell_id = api.submit_order(symbol, qty, side='sell', order_type='market', time_in_force='day')
    else:
        price = place_limit_with_padding(api, symbol, 'sell', pad_pct=0.01)
        logger.info(f"Placing LIMIT SELL {qty} {symbol} @ {price} (extended hours)")
        sell_id = api.submit_order(symbol, qty, side='sell', order_type='limit', time_in_force='day', limit_price=price, extended_hours=True)

    if not sell_id:
        raise SystemExit("Failed to submit SELL order")

    info = api.wait_for_fill(sell_id, timeout_seconds=90 if is_open else 10)
    logger.info(f"SELL status: {info}")
    if not info or info.get('status') != 'filled':
        if not is_open:
            raise SystemExit("Sell not filled (market closed). Re-run during market hours for guaranteed fill.")
        raise SystemExit("Sell did not fill in time.")

    logger.success("Test trade complete: BUY then SELL filled.")


if __name__ == '__main__':
    main()


