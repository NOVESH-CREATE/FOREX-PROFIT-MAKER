# 📋 ORB STRATEGY — QUICK CHEAT SHEET (everything you forgot)

**What it is:** An **Opening Range Breakout (ORB)** system. Every day you mark the High & Low of a special
**15-minute "mother" candle**, then you only trade when price *breaks out* of that range. One trade per pair per day.
6 setups per week (Tuesday = NO TRADE).

---

## 1) The weekly schedule — the whole strategy in one table

All times you'll see in your app were stored in **GMT** and shown in **IST** (India, GMT+5:30). Your MT5 export time is the **broker server time** — it matched GMT in your data files.

| # | Day | Pair 💱 | ORB window (IST) | ORB window (GMT/server) | Mother candle TF | Breakout confirmed on | RR (Reward:Risk) | Expected WR* |
|---|-----|---------|------------------|--------------------------|------------------|------------------------|------------------|--------------|
| 1 | **Monday** | **USDCAD** | 09:15 PM | 15:45 | M15 | M15 | **1 : 3** | 61.5% |
| 2 | **Tuesday** | ❌ nothing | — | — | — | — | — | (no setup ≥60%) |
| 3 | **Wednesday** | **GBPUSD** | 08:30 PM | 15:00 | M15 | M15 | **1 : 3** | 61.5% |
| 4 | **Thursday** | **USDCAD** | 04:45 PM | 11:15 | M15 | M15 | **1 : 2** | 72.7% |
| 5 | **Thursday** | **EURUSD** | 05:00 PM | 11:30 | M15 | **M5** ⚡ | **1 : 2** | 66.7% |
| 6 | **Thursday** | **GBPUSD** | 09:15 AM | 03:45 | M15 | M15 | **1 : 2** | 63.6% |
| 7 | **Friday** | **USDCAD** | 09:30 PM | 16:00 | M15 | M15 | **1 : 2.5** | 63.0% |

*Expected WR = the win rate measured in your own 6-month backtest (that's how each slot earned its place).

So in plain words:

- **Mother candle timeframe = M15 for ALL 6 setups** — always look at the 15-min candle that opens at the listed time, mark its **High** and **Low**.
- **Breakout timeframe = M15 for 5 setups**, but the **Thursday EURUSD 05:00 PM** setup waits for the breakout on the **M5** chart instead.
- **Stop loss** = opposite side of the mother candle + **2 pips** buffer.
- **Take profit** = your risk (SL distance) × **RR** from the table above.
- **Thursday is your busy day**: 3 setups (one per pair, so 3 possible trades). Mon/Wed/Fri = 1 setup each.

---

## 2) The rules, step by step (exactly as coded in `app.py`)

1. **Wait for the setup's time.** Example: Monday 21:15 IST → open the USDCAD M15 chart.
2. **Draw the ORB range.** The 15-minute candle *starting at that exact time* is the "mother candle". Note its **HIGH** and **LOW**. Its range must be between **2 and 100 pips**, otherwise skip that day (too tight/too wide = no trade).
3. **Wait for a confirmed breakout.** After the mother candle, wait for a candle to **CLOSE above the High** (you go **LONG**) or **CLOSE below the Low** (you go **SHORT**). (For EURUSD-Thursday use the M5 chart for this wait.)
4. **Enter** at the **close** of that breakout candle.
5. **Stop Loss** = the opposite side of the mother candle **minus/plus 2 pips**:
   - LONG → SL just below the Low − 2 pips
   - SHORT → SL just above the High + 2 pips
6. **Take Profit** = (entry − SL distance) × **RR** of that slot.
7. **Only the FIRST breakout** of the day counts — no re-entries after a failed trade.
8. If neither TP nor SL is hit, the trade is closed at the **end of the trading day**.
9. **Max 1 trade per pair per day** (Thursday can still have up to 3 — one on each pair).

---

## 3) Your backtest scoreboard (so you remember how it did)

| Metric | Result |
|---|---|
| Test window | 1 Aug 2025 → 30 Jan 2026 (131 trading days) |
| Trades taken | 136 (≈5 per week) |
| Win rate | **64.7%** (88 wins / 48 losses) |
| Average reward:risk used | 1 : 2.48 |
| Edge per trade | **+1.24R** |
| Total | **+168.5R** → at $25,000 & $200/trade risk: **+$33,700 (+134.8%)** |
| Max losing streak | 4 (that's your worst case = only −$800 on $25k) |
| Losing months | none |

Per-slot in-sample record (this is where the "expected WR" came from):

| Slot | Trades | Win rate |
|---|---|---|
| Mon USDCAD 9:15 PM | 26 | 61.5% |
| Wed GBPUSD 8:30 PM | 26 | 61.5% |
| Thu USDCAD 4:45 PM | 22 | **72.7%** |
| Thu EURUSD 5:00 PM | 24 | 66.7% |
| Thu GBPUSD 9:15 AM | 11 | 63.6% |
| Fri USDCAD 9:30 PM | 27 | 63.0% |

---

## 4) ⚠️ Two data-quality notes you MUST remember before forward-testing

1. **Your old EURUSD & GBPUSD CSV files were mislabeled** (the file called "EURUSD M5" actually contains 15-minute candles, and the file called "GBPUSD M15" actually contains 5-minute candles). Your Feb-8 backtest was computed on those files as they were. **When you export the new Feb→Sep 2026 files, export TRUE M5 and TRUE M15** — and the harness I built will verify the spacing automatically.
2. Backtest numbers contain **no spread/commission/slippage**, and the setups were *chosen* because they were winners in that same window — so expect the forward result to be a bit worse than 64.7% / +1.24R. Break-even win rate for RR 2.5 is only ~29%, so there is a large safety margin.

---

## 5) To run the forward test (Feb → Sep 2026)

Fetch **6 files** from your broker's MT5 (export as CSV/TXT, same tab-format as the old ones):
EURUSD, GBPUSD, USDCAD — **one M5 + one M15 each**, covering ≈ 1 Feb 2026 → 8 Sep 2026.

Put them in a folder (e.g. `forward_data/`) in this repo and run:

```bash
python3 strategy_analysis/run_forward_test.py --data forward_data
```

It checks the timeframes, runs the exact same engine on the new 7 months, and gives you a backtest-vs-forward comparison (trades, win rate, net R, profit factor, drawdown, per-slot win rates).

If you prefer, just **upload the 6 files here** and I will run it for you.
