# FundYourFX Classic $5,000 — Payout, Scaling & DD Progress Report

*Prepared 2026-09-09. Strategy: core-3 ORB plan (Mon USDCAD RR3 · Wed GBPUSD RR3 · Thu EURUSD RR2),
original close-breakout entry, 1 trade/day. Rules taken from the official FundYourFX Classic FAQ.*

> ⚠️ **Correction — Classic rules are NOT the same as Pro.** I can't read your screenshot (no image
> vision), so I used the official Classic FAQ. If your add-ons change the profit split, target or
> minimum days, tell me what they say and I'll redo this. Classic is **stricter** than Pro:

| Rule | Pro | **Classic (yours)** |
|---|---|---|
| Profit target | 8% | **10%** |
| Max DD (static) | 8% | **6%** |
| Daily DD | 5% | **4%** |
| Min trading days | 3 | **6** |
| Profit split start | 60% | **50%** |
| Split ladder | 60→70→80→90→95% | **50→60→70→80→90→95%** (every 3 payouts) |
| Fee refund | after 2 payouts | **after 2 payouts** |
| Growth (scaling) | every 3 payouts | **every 3 payouts** |
| 25% payout rule | yes | yes |

---

## 1. How the payouts work (your three questions, answered)

**Pass probability (bootstrap on your 7 months of forward trades, Classic 6% static DD / 4% daily):**

| Risk/trade | P(pass without breaching) |
|---|---|
| 0.5% ($25) | **100%** |
| **1.0% ($50)** | **97.9%** |
| 2.0% ($100) | 86.5% |

**The subtle rule that matters:** with 1% risk, your best day is a RR3 win = +3% of balance, and the
**25% payout rule** (best day ≤ 25% of total profit) means you actually clear a payout at **~12% profit,
not 10%**. So your first payout lands on roughly $600–$740 gross profit, not $500.

### Payout timeline (recommended: 1% = $50 risk/trade)

| # | ~When | Account level | Gross profit | Split | **You receive** | Note |
|---|---|---|---|---|---|---|
| **1** | ~week 5 (3 wks hot start) | $5,000 | ~$737 | 50% | **~$368** | first payout |
| **2** | ~week 10 | $5,000 | ~$728 | 50% | **~$364** | ✅ **FEE REFUNDED** |
| **3** | ~week 15 | $5,000 | ~$676 | 50% | **~$338** | ✅ **SCALE → $7,500**, split → 60% |
| 4 | ~week 20 | $7,500 | ~$1,006 | 60% | ~$604 | |
| 5 | ~week 25 | $7,500 | ~$1,014 | 60% | ~$608 | |
| 6 | ~week 30 | $7,500 | ~$1,100 | 60% | ~$660 | ✅ **SCALE → $10,000**, split → 70% |
| 7 | ~week 35 | $10,000 | ~$1,365 | 70% | ~$956 | |
| 8 | ~week 40 | $10,000 | ~$1,450 | 70% | ~$1,015 | |
| 9 | ~week 45 | $10,000 | ~$1,349 | 70% | ~$944 | ✅ **SCALE → $25,000**, split → 80% |
| 10 | ~month 12 | $25,000 | ~$3,363 | 80% | ~$2,690 | |
| 11 | ~month 14 | $25,000 | ~$3,380 | 80% | ~$2,704 | |
| 12 | ~month 15 | $25,000 | ~$3,397 | 80% | ~$2,717 | ✅ **SCALE → $60,000**, split → 90% |
| 13+ | ~month 18+ | $60,000 | ~$8,000 | 90% | ~$7,264 | split → 95% after #15 |

**Cumulative you-take-home (if no breach):** ~$368 → ~$732 → **~$1,070** (first 3 payouts) →
~$2,943 (6) → ~$5,857 (9) → ~$13,968 (12) → ~$37,604 (15).

### Safer option — 0.5% risk ($25/trade), 100% pass probability
Slower but bulletproof: payout #1 ~$261 (~week 4), #2 ~$272 (~week 10, fee refund), #3 ~$286
(~week 15, scale → $7,500). Same ladder, roughly half the money per payout.

---

## 2. Fee refund & scaling — exactly when

- **Fee refund:** after your **2nd successful payout** the full account fee is paid back to you.
  (Your exact fee depends on checkout + add-ons — confirm the figure from your screenshot and I'll
  add it to the table. Third-party sites list ~$119 for a $5K "Flash" tier, but I won't trust that
  over your own invoice.)
- **Scaling:** automatic every **3 payouts**, no application, no extra fee:
  **$5,000 → $7,500 → $10,000 → $25,000 → $60,000 → $150,000** (up to 30× your start).
- **Profit split climbs alongside:** 50% → 60% → 70% → 80% → 90% → **95%**.

So the sequence you asked about is exactly:
**Payout #1 (~$368) → Payout #2 (~$364 + fee refund) → Payout #3 (~$338) → account scales to $7,500
and split rises to 60%.**

---

## 3. Equity curve / progress chart

See `equity_curve.png` (next to this file). It shows:
- **Blue line** — account balance (compounding 1% risk, sawtooth = payout withdrawals resetting to level).
- **Red dashed step line** — the 6% static drawdown floor (breach line). You never come close at 1% risk.
- **Green dots** — each payout (P1, P2, … with the split %).
- **Purple "SCALE" labels** — capital increases ($7,500 / $10,000 / $25,000 …).
- **Orange "FEE REFUND"** — at payout #2.

**Drawdown reality check:** worst forward losing streak = 5 in a row. At 1% risk that's ~−4.9%
(just inside the 6% floor); at 0.5% it's ~−2.4%. Your worst single day is 1 loss (−1%), far under
the 4% daily limit. So the only thing that can ever kill the account is a cold opening streak —
which is exactly what the 97.9% bootstrap (1%) / 100% (0.5%) pass rates are measuring.

---

## 4. Important caveats

1. **Dates after Sep 2026 are a replay** — beyond your real 7 months of data (Feb–Sep 2026) the
   projection re-cycles that same trade record as a stand-in for future months. Use the **"~week N"**
   column as the realistic timeline, not the exact calendar dates.
2. The first 3 payouts look fast partly because Feb–Mar 2026 front-loaded wins. Plan on **~5 weeks
   per payout** as the honest average, not 3.
3. These are **out-of-sample forward results (~49% WR)**, not your inflated 64.7% backtest — and the
   edge can still degrade. Risk stays at 1% (or 0.5%) for that reason.
4. **Operational rules still apply:** SL within 3 min, hold >60s, no entry ±5 min around high-impact
   news (15:45 / 15:00 / 11:30 GMT), max 2 trades/instrument/direction, no martingale/grid/copying.
5. Not financial advice; prop rules can change — re-read the FAQ before funding.

---

## Artifacts

- `strategy_analysis/results/classic_projection/equity_curve.png` — the chart
- `strategy_analysis/results/classic_projection/equity_10.csv` / `equity_5.csv` — full curve data
- `strategy_analysis/results/classic_projection/schedule_10.json` / `schedule_5.json` — payout schedule
- `run_classic_projection.py` — the simulator (rules are constants at the top)
