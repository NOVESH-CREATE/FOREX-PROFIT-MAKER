# FOREX-PROFIT-MAKER — 3-Setup Funded Plan (Classic $5K + add-ons)

**The plan:** trade ONLY 3 ORB setups per week on a FundYourFX **Classic $5,000**
account with add-ons (**90% profit split** from day one, **8% max loss**), and
reinvest every fee refund into a **new $5K account** that trades the same
signals (multi-account growth). No other setups, no extra entries.

## The core-3 setups (GMT times, 2-pip SL buffer)

| Day | Pair | ORB candle (GMT) | Entry | RR |
|---|---|---|---|---|
| Monday | USDCAD | 15:45 | 15-min close-breakout | 1:3 |
| Wednesday | GBPUSD | 15:00 | 15-min close-breakout | 1:3 |
| Thursday | EURUSD | 11:30 | 5-min close-breakout | 1:2 |

Risk is always **1% of the account's level** per trade — it never increases by
choice; a scaled account's dollar risk grows only because its funded level grew.

## Funded-account rules modeled (Classic + add-ons)

- **10%** profit target and **min 6 trading days** before a payout
- **8% static max loss** (hard breach — account dead)
- **4% daily drawdown** (soft breach, monitored)
- **25% rule** — best single day ≤ 25% of the run's total profit
- **90% profit split** on every payout
- **Fee refund** after an account's **2nd payout** → buys a NEW $5K account
- Per-account **scaling every 3 payouts**: $5K → $7.5K → $10K → $25K → $60K → $150K

## Run

```bash
python3 funded_plan.py        # needs: pandas, numpy, plotly
```

The script is fully self-contained except for the trade engine
(`strategy_analysis/backtest_engine.py`, which it imports for trade execution).
It parses the 4 keeper MT5 **.htm** history exports in the repo root
(UTF-16/UTF-8 aware), generates the core-3 signals for **1 Feb → 9 Sep 2026**
(no future replay), simulates the funded accounts, and writes the deliverable
into `strategy_analysis/results/multiacct/`:

- `multiacct_equity.html` / `index.html` — interactive **3-panel Plotly chart**
  (combined equity vs the 8% max-loss floor with per-account dotted equity ·
  cumulative payouts · active accounts), full hover labels
- `multiacct_report.md` — account-by-account report + payout ledger
- `multiacct_summary.json` — machine-readable summary
- `combined_equity.csv`, `account_<n>_equity.csv` — equity series

**Verified output (1% risk):** 4 active accounts · combined funded **$27,500** ·
combined balance **$28,025** · total paid out **$7,965** by 9 Sep 2026.

## Repo layout

- `funded_plan.py` — the one script the plan needs (simulation + chart + report)
- `strategy_analysis/backtest_engine.py` — ORB trade engine (kept, imported by `funded_plan.py`)
- `*.htm` — the 4 keeper MT5 history exports (EURUSD M15/M5, GBPUSD M15, USDCAD M15)
- `strategy_analysis/results/multiacct/` — final deliverable (interactive chart + report)
- `strategy_analysis/results/classic_addon/` — Classic add-on payout-table reference
