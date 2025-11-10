import argparse
import time
from pathlib import Path

import pandas as pd

from src.python.broker.alpaca_client import get_historical_crypto_df, load_secrets, AlpacaClient
from src.python.environments.trading_env import TradingEnv, TradingEnvConfig
from src.python.drl.dqn_agent import DQNAgent, DQNConfig
from src.python.reports.reporting import save_csvs, ReportConfig


def run_paper(symbol: str, ckpt: str, interval: str = "1Min", window: int = 60, start: str = None, end: str = None, execute: bool = False):
    df = get_historical_crypto_df(symbol, start, end, timeframe=interval)
    if df.empty:
        print("No data returned.")
        return
    env = TradingEnv(df, TradingEnvConfig(window=window, initial_cash=50.0))
    obs_dim = len(env.reset())
    agent = DQNAgent(obs_dim, 3, DQNConfig(device="cpu"))
    agent.load(ckpt)
    agent.epsilon = 0.0
    eq_series = []
    trades = []
    obs = env.reset()
    secrets = load_secrets()
    client = AlpacaClient(secrets) if (execute and secrets) else None
    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        eq_series.append((info["time"], info["equity"]))
        # Optional execution (market orders) - demonstration only, not HFT safe
        if client and action in (1, 2):
            side = "buy" if action == 1 else "sell"
            try:
                client.submit_market_order_crypto(symbol, notional=5.0, side=side)
            except Exception:
                pass
        obs = next_obs
        if done:
            break
        # simulate 1m cadence if desired
        # time.sleep(60)
    equity = pd.Series([e for _, e in eq_series], index=[t for t, _ in eq_series], name="equity")
    out = Path("results/paper")
    out.mkdir(parents=True, exist_ok=True)
    save_csvs(ReportConfig(out_dir=str(out)), equity, pd.DataFrame(), None)
    print(f"Paper run complete. Equity CSV at {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--interval", default="1Min")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    run_paper(args.symbol, args.ckpt, args.interval, args.window, args.start, args.end, args.execute)

import argparse
import time
from pathlib import Path
import pandas as pd

from src.python.environments.trading_env import TradingEnv, TradingEnvConfig
from src.python.drl.dqn_agent import DQNAgent, DQNConfig
from src.python.broker.alpaca_client import AlpacaConfig, fetch_crypto_bars, submit_market_order, get_rest
from src.python.metrics.metrics import summarize

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def load_secrets():
    cfg_path = Path("config/secrets.yaml")
    if yaml is None or not cfg_path.exists():
        return None
    data = yaml.safe_load(cfg_path.read_text())
    return AlpacaConfig(
        api_key=data["ALPACA_API_KEY_ID"],
        api_secret=data["ALPACA_API_SECRET_KEY"],
        base_url=data["ALPACA_PAPER_BASE_URL"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--dry-run", action="store_true", help="do not place orders")
    args = p.parse_args()

    secrets = load_secrets()
    if secrets is None:
        raise SystemExit("Missing config/secrets.yaml or PyYAML.")

    agent = DQNAgent(input_dim=args.window * 6, n_actions=3, cfg=DQNConfig(epsilon_start=0.0, epsilon_end=0.0))
    agent.load(args.ckpt, map_location="cpu")
    print("Loaded policy:", args.ckpt)

    last_ts = None
    equity_curve = []
    position = 0
    entry_px = None
    cash = 50.0

    while True:
        # fetch last ~2 hours to build window
        api = get_rest(secrets)
        end = pd.Timestamp.utcnow()
        start = (end - pd.Timedelta(minutes=180)).isoformat()
        end_iso = end.isoformat()
        df = fetch_crypto_bars(secrets, args.symbol, start=start, end=end_iso, timeframe="1Min")
        if df.empty or len(df) < args.window + 1:
            time.sleep(args.poll)
            continue
        env = TradingEnv(df, TradingEnvConfig(window=args.window, initial_cash=cash))
        s = env.reset(start_index=len(env.feat_df) - 1)
        a = agent.act(s)
        # map action to orders
        row = env.feat_df.iloc[-1]
        open_px = float(row["Open"])
        if a == 1 and position == 0:
            qty = max(0.0, cash // open_px)
            if qty >= 1:
                if not args.dry_run:
                    submit_market_order(secrets, args.symbol, qty, "buy")
                position = 1
                entry_px = open_px
                cash -= qty * open_px
                print(f"BUY {qty} @ {open_px}")
        elif a == 2 and position == 1 and entry_px is not None:
            qty = 1  # close unit size (simplified)
            if not args.dry_run:
                submit_market_order(secrets, args.symbol, qty, "sell")
            pnl = open_px - entry_px
            cash += entry_px + pnl
            position = 0
            entry_px = None
            print(f"SELL {qty} @ {open_px}  pnl={pnl:.2f}")
        equity = cash + (open_px - (entry_px or open_px))
        equity_curve.append(equity)
        print(f"Equity={equity:.2f}  Pos={position}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()


