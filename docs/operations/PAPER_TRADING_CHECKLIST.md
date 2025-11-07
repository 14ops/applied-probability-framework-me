# ✅ Paper Trading Readiness Checklist

Complete checklist to ensure your Stock AI framework is ready for paper trading on Alpaca.

---

## 🔧 Pre-Launch Setup

### 1. **Alpaca API Connection** ✅
- [x] `stock_ai/api/alpaca_interface.py` exists with full API wrapper
- [x] Methods available: `get_account()`, `submit_order()`, `get_position()`, `get_quote()`, `get_current_price()`
- [ ] **ACTION:** Verify API connection works:
  ```bash
  python -c "from stock_ai.api.alpaca_interface import AlpacaInterface; api = AlpacaInterface(); print(api.get_account())"
  ```

### 2. **Secrets Configuration** ✅
- [x] Template exists: `config/secrets.yaml.template`
- [ ] **ACTION:** Create `config/secrets.yaml` with your paper trading keys:
  ```yaml
  alpaca:
    key_id: "YOUR_PAPER_KEY_ID"
    secret_key: "YOUR_PAPER_SECRET_KEY"
    base_url: "https://paper-api.alpaca.markets"
  ```
- [ ] **VERIFY:** File is in `.gitignore` (should not be committed)

### 3. **Live Runner** ✅
- [x] `stock_ai/live_trading/live_runner.py` exists with complete implementation
- [x] Includes risk management (drawdown limits, consecutive losses)
- [x] Includes kill switch file checking
- [x] Includes comprehensive trade logging
- [x] Includes market hours detection
- [ ] **ACTION:** Review risk parameters in `config/settings.yaml`

### 4. **Risk Management** ✅
- [x] Max drawdown protection (default: 5%)
- [x] Consecutive loss protection (default: 5 losses)
- [x] Kill switch file (`stop.txt`) support
- [x] Position sizing limits
- [ ] **ACTION:** Configure in `config/settings.yaml`:
  ```yaml
  risk:
    max_drawdown_pct: 5.0
    max_consecutive_losses: 5
    kill_switch_file: "stop.txt"
  ```

### 5. **Logging & Reporting** ✅
- [x] Rotating log files (daily, compressed)
- [x] Trade logging to `results/trades_log.csv`
- [x] Equity history logging
- [x] Per-date log files in `logs/` directory
- [ ] **VERIFY:** `logs/` and `results/` directories exist

### 6. **Backtest Validation** ⬜
- [ ] **ACTION:** Run verification backtest:
  ```bash
  python run_backtest.py
  ```
- [ ] **VERIFY:** Backtest completes successfully
- [ ] **VERIFY:** Strategy matches what you'll use for live trading

### 7. **Configuration Review** ✅
- [x] `config/settings.yaml` includes all required sections
- [x] Strategy parameters match backtest
- [x] Trading symbols configured
- [ ] **ACTION:** Review and adjust:
  - `tickers`: List of symbols to trade
  - `strategy.name`: Strategy to use
  - `strategy.params`: Strategy parameters
  - `live.check_interval`: Seconds between checks (default: 300 = 5 minutes)

---

## 🚀 Launch Steps

### Step 1: Verify Secrets
```bash
# Test Alpaca connection
python -c "from stock_ai.api.alpaca_interface import AlpacaInterface; api = AlpacaInterface(); print('✅ Connected!', api.get_account())"
```

### Step 2: Run Verification Backtest
```bash
python run_backtest.py
```
Expected: Should complete and show metrics without errors.

### Step 3: Review Configuration
```bash
# View current config
cat config/settings.yaml

# Verify secrets template (don't view actual secrets!)
cat config/secrets.yaml.template
```

### Step 4: Start Paper Trading
```bash
# From project root
python stock_ai/live_trading/live_runner.py
```

Or from within `stock_ai` directory:
```bash
cd stock_ai
python live_trading/live_runner.py
```

---

## 🛡️ Safety Features (Auto-Enabled)

The live runner includes these safety features that activate automatically:

1. **Kill Switch**: Create `stop.txt` file in project root to immediately stop trading
2. **Max Drawdown**: Trading stops if drawdown exceeds configured limit (default: 5%)
3. **Consecutive Losses**: Trading stops after N consecutive losses (default: 5)
4. **Market Hours**: Only trades during market hours
5. **Position Limits**: Respects `max_position_pct` from config

---

## 📊 Monitoring While Running

### Check Logs
```bash
# View latest log
tail -f logs/live_trading_$(date +%Y-%m-%d).log

# View trade log
cat results/trades_log.csv
```

### Check Status
```bash
# Kill switch status
ls -la stop.txt  # If exists, trading will stop

# Check equity history
cat logs/equity_history_$(date +%Y%m%d).csv
```

### Stop Trading
```bash
# Method 1: Kill switch file (recommended)
touch stop.txt

# Method 2: Ctrl+C (graceful shutdown)
```

---

## 🔍 Troubleshooting

### Issue: "Secrets file not found"
**Solution:** Copy template:
```bash
cp config/secrets.yaml.template config/secrets.yaml
# Then edit with your keys
```

### Issue: "Alpaca API connection failed"
**Solution:**
1. Verify keys are correct
2. Check `base_url` is `https://paper-api.alpaca.markets` (not live)
3. Verify API keys are active in Alpaca dashboard

### Issue: "No data for symbol"
**Solution:**
1. Verify symbol is valid (e.g., AAPL, not APPL)
2. Check internet connection
3. Try different symbol in config

### Issue: "Market is closed"
**Solution:** This is normal outside market hours. Bot will wait and check periodically.

---

## 📈 Post-Launch

### Daily Review
1. Check `results/trades_log.csv` for trade history
2. Review `logs/equity_history_*.csv` for equity curve
3. Monitor drawdown percentage
4. Review consecutive losses count

### Weekly Review
1. Analyze strategy performance
2. Compare to backtest results
3. Adjust parameters if needed
4. Review risk limits

---

## ✅ Final Pre-Launch Checklist

- [ ] Secrets file created with paper trading keys
- [ ] Secrets file verified with test connection
- [ ] Backtest run successfully
- [ ] Risk parameters configured appropriately
- [ ] Strategy parameters match backtest
- [ ] Trading symbols verified
- [ ] `logs/` and `results/` directories exist
- [ ] Understand kill switch operation (`stop.txt`)
- [ ] Know how to monitor logs
- [ ] Paper trading URL confirmed (not live!)

---

## 🎯 Ready to Launch?

Once all boxes are checked:

```bash
# Start paper trading
python stock_ai/live_trading/live_runner.py
```

The bot will:
1. Connect to Alpaca
2. Show portfolio summary
3. Ask for confirmation
4. Begin trading cycles

**Remember:** This is PAPER TRADING. No real money at risk, but always verify `base_url` is the paper trading URL!

---

**Last Updated:** 2024-01-XX  
**Status:** ✅ All Components Complete - Ready for Testing

