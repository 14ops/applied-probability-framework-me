# 📈 Applied Probability & Automation Framework for High-RTP Games and Algorithmic Trading

This framework is a comprehensive research and engineering project designed to model, simulate, and automate strategies across high-Return-to-Player (RTP) probabilistic systems, including casino-style games and financial markets.

The project has evolved from a focus on high-frequency, low-edge games (like Mines) to a full-fledged **Algorithmic Trading MVP** to apply the same principles of expected value (EV) analysis and risk management to equities.

## 🎯 Project Goals

1.  **Statistical Modeling:** Accurately model the probability and expected value (EV) of complex, multi-step probabilistic systems.
2.  **Strategy Automation:** Implement and backtest various decision-making strategies, personified as "Character Strategies" (e.g., Takeshi, Aoi).
3.  **Risk Management:** Integrate advanced bankroll management, stop-loss, and profit-taking logic to maximize long-term growth and minimize risk of ruin.
4.  **Algorithmic Trading MVP:** Pivot the core logic to a financial backtesting environment to test strategies on real-world market data.

## 🚀 Equities MVP: Algorithmic Trading Components

The core framework has been extended to support algorithmic trading backtesting, allowing strategies to be tested against historical stock data.

| Component | File Path | Description |
| :--- | :--- | :--- |
| **Trading Environment** | `src/python/environments/equities_env.py` | The core backtesting engine, handling trade execution, slippage, commissions, and PnL calculation. |
| **Data Loader** | `src/python/data/equities_loader.py` | Handles loading and cleaning historical OHLCV data for backtesting. |
| **Technical Indicators** | `src/python/indicators/ta.py` | Pure NumPy/Pandas implementation of common indicators (SMA, EMA, RSI, ATR, Bollinger Bands). |
| **Character Strategies** | `src/python/strategies/*.py` | Strategies adapted to return trading actions (`buy`, `sell`, `hold`) instead of game actions. |

## 📊 Technical Analysis Indicators (`ta.py`)

The following indicators are implemented to provide market signals for the trading strategies:

| Indicator | Purpose | Key Feature |
| :--- | :--- | :--- |
| **RSI(14)** | Momentum | Uses Wilder's smoothing (EWM) for accurate calculation. |
| **SMA/EMA** | Trend Following | Simple and Exponential Moving Averages for trend identification. |
| **ATR(14)** | Volatility/Risk | Used for dynamic stop-loss and profit-taking logic. |
| **Bollinger Bands** | Volatility/Reversion | Measures market volatility and potential mean-reversion points. |

## 🧠 Character Strategies (The "Anime Arc")

Each strategy is personified to represent a distinct risk profile and decision-making logic.

| Character | Archetype | Core Logic |
| :--- | :--- | :--- |
| **Takeshi** | Aggressive Berserker | High-momentum, high-risk, high-reward plays. |
| **Aoi** | Mean Reversion Specialist | Bets on market overextensions returning to the mean. |
| **Yuzu** | Volatility Breakout | Trades based on significant price movements breaking out of consolidation. |
| **Kazuya** | Defensive Trend Follower | Slow, steady, capital-preserving strategy. |
| **Lelouch** | Meta-Strategist | Adaptive logic that switches between other strategies based on market regime. |

## 🎲 Mines Game Probability Analysis (Legacy)

The original core of the framework, focused on modeling the Mines game (25 tiles, 2 mines). This section remains for validation of the core probability engine.

| Clicks | Payout (est) | Win Probability | Expected Value (EV) |
| :--- | :--- | :--- | :--- |
| 1 | 1.04x | 92.0% | 0.9568 |
| 2 | 1.14x | 84.3% | 0.9610 |
| 3 | 1.25x | 77.0% | 0.9625 |
| 4 | 1.37x | 70.0% | 0.9590 |
| 5 | 1.52x | 63.3% | 0.9622 |
| 6 | 1.68x | 57.0% | 0.9576 |
| 7 | 1.88x | 51.0% | 0.9588 |
| 8 | 2.12x | 45.3% | 0.9593 |
| 9 | 2.40x | 40.0% | 0.9600 |
| 10 | 2.74x | 35.0% | 0.9590 |

## 🧪 Testing and Validation
### Automated Testing
```bash
# Run all tests
python -m pytest tests/
# Run specific test categories
python -m pytest tests/test_math.py
python -m pytest tests/test_strategies_*.py
```
### Performance Benchmarking
```bash
# Run comprehensive Mines benchmark
python examples/run_mines_benchmark.py
# Run Equities benchmark (MVP)
python src/python/examples/run_equities_benchmark.py
```

## 📈 Advanced Features
### 1. Deep Reinforcement Learning (DRL)
- **Application**: Training a DQN agent to make optimal `buy/sell/hold` decisions in the equities environment.
- **Integration**: Seamless integration with existing strategies for hybrid decision-making.

### 2. Bayesian Risk Management
- **Method**: Probabilistic modeling to estimate the uncertainty of strategy outcomes and the probability of drawdown.
- **Feature**: Dynamic bet sizing (Kelly Criterion) and stop-loss based on real-time risk assessment.

### 3. Multi-Agent Consensus
- **Feature**: Implementing a voting system where multiple strategies (characters) must agree on a trade action, reducing single-strategy bias.

## 📚 Documentation
- [API Reference](docs/API_REFERENCE.md)
- [Strategy Guide](docs/guides/strategy_formulas.md)
- [Implementation Guide](docs/guides/implementation_guide.md)

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- **Research Foundation**: Based on comprehensive analysis of high-RTP games and quantitative finance principles.
- **Character Strategies**: Inspired by popular anime and manga characters.

---
**⚠️ Disclaimer**: This framework is for educational and research purposes only. It is not financial advice. Algorithmic trading involves significant risk, and past performance is not indicative of future results. Please invest responsibly and within your means.
