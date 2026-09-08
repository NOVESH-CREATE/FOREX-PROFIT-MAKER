# FundYourFX Instant Funding (Pro) — How to pass without breaching

*Prepared 2026-09-09 · All numbers are OUT-OF-SAMPLE forward trades (Feb 2 → Sep 7 2026)
from your uploaded .htm files, run through the same ORB engine.*

---

## 1. The rules that actually bind (from FundYourFX's FAQ)

| Rule | Value | Breach type |
|---|---|---|
| Max Drawdown (static) | **8%** from initial balance | **HARD** (account closed) |
| Daily Drawdown | **5%** of prior end-of-day balance (resets 22:00 UTC) | soft |
| Profit target (payout) | **8%** | — |
| Min trading days | 3 | — |
| Max time | unlimited | — |
| 25% rule | best single day ≤ 25% of total profit | delays payout only |
| Stop loss | must be set within 3 min | soft |
| Min hold | 60 sec | soft |
| Stacking | max 2 trades / instrument / direction | **HARD** |
| News | no entry ±5 min around high-impact news | soft |
| Prohibited | martingale, grid, HFT, arbitrage, scalping, copy | **HARD** |

Your strategy already satisfies SL-in-3-min (bracket order), 60-sec hold, stacking (≤1 trade/instrument/day) and is not martingale/grid — so the binding rules are **8% static DD, 5% daily DD, 8% target, 25% rule**.

---

## 2. Direct answers to your three questions

### Q1 — All ORBs or a single ORB? → **Neither "all" nor "one": trade 3 core setups, drop 2.**

The forward test splits your 6 setups into clear winners and losers:

| Setup | n | Win rate | Net R | Verdict |
|---|---|---|---|---|
| Wednesday GBPUSD 08:30 PM (RR3) | 31 | 48.4% | **+29** | ✅ KEEP (best) |
| Monday USDCAD 09:15 PM (RR3) | 32 | 46.9% | **+28** | ✅ KEEP |
| Thursday EURUSD 05:00 PM (RR2) | 31 | 51.6% | **+17** | ✅ KEEP |
| Thursday USDCAD 04:45 PM (RR2) | 31 | 41.9% | +8 | ⚪ optional |
| Friday USDCAD 09:30 PM (RR2.5) | 30 | 36.7% | +8.5 | ⚪ weak |
| Thursday GBPUSD 09:15 AM (RR2) | 30 | 30.0% | **−3** | ❌ DROP (losing) |

Trading **all 6** also stacks 3 trades every Thursday, so one bad Thursday = −3R in a single
day — that's what pushes your daily drawdown. The core 3 are on **different days** (Mon/Wed/Thu),
so your worst day is only −1R.

### Q2 — With retest or without? → **Without (original close-breakout entry).**

The limit-retest idea is neutral-to-negative overall on forward data:

- All setups: original +87.5R vs retest +79.5R (same PF 1.83).
- It helps only Wednesday GBPUSD (+35R vs +29R) and Thursday USDCAD (+11R vs +8R).
- It **hurts** Monday USDCAD badly (+16R vs +28R) and Friday USDCAD (+4.5R vs +8.5R).

So the simpler original entry wins. (If you want, run retest **only** on Wednesday GBPUSD.)

### Q3 — Reduce RR from 1:3 to 1:2 or 1:1? → **NO. Keep 1:3 (and 1:2 where it already is).**

Re-simulating the full set at uniform RR (forward data):

| RR | Trades | Win rate | Net R | PF |
|---|---|---|---|---|
| 1:1 | 185 | 53.0% | **+11** | 1.13 |
| 1:2 | 185 | 42.7% | +52 | 1.49 |
| 1:3 | 185 | 39.5% | **+107** | 1.96 |
| **native (3/2/2.5)** | 185 | 42.7% | +87.5 | 1.83 |

Lowering the RR raises your win rate a bit but **cuts the edge by 50–90%** — at 1:1 the strategy
barely makes money and even trips the 25% payout rule. The RR3 setups (Monday USDCAD, Wednesday
GBPUSD) are where the profit comes from; do not clip their targets.

---

## 3. The recommended plan + the funded-account math

**Trade:** Monday USDCAD 09:15 PM (RR3) · Wednesday GBPUSD 08:30 PM (RR3) · Thursday EURUSD 05:00 PM (RR2)
— original entry, no retest. *(Optional 4th: Thursday USDCAD 04:45 PM, RR2.)*

| Risk / trade | Reaches +8% in | Max DD (static) | Daily breaches | Bootstrap pass prob |
|---|---|---|---|---|
| 0.5% | ~10 trades (~3 wks) | ~0.5% | 0 | **100%** |
| **1.0%** | **~7 trades (~2 wks)** | **~1.0%** | **0** | **99.2%** |
| 2.0% | ~5 trades | ~2.0% | 0 | 92.6% |

(With the optional 4th setup: 1% risk → 10 trades, max DD ~1%, pass prob 98.0%.)

**Sizing rule of thumb that guarantees no daily breach:**
your worst day is 1 loss (−1R) on the core-3 plan, so **risk 1% per trade = max 1% daily loss** —
a huge margin under the 5% daily limit, and a 9-loss streak would still only be ~9%… so keep
1% (or 0.5% for maximum safety). Anything above ~2.5% starts eating into your daily-drawdown
buffer, and the old 10%-per-trade compounding from the original app would breach an 8% account
almost immediately.

**25% payout rule:** best single day on the core-3 plan is ~4% of total profit — comfortably compliant.

**Pass probability interpretation:** bootstrap resamples your 7 months of actual forward trades
(day-blocks) 2,000×; "99.2%" = in 99.2% of reshuffled histories you hit +8% before a −8% hard
breach, with zero daily soft breaches. That's a sample-based confidence estimate, not a guarantee.

---

## 4. Honest caveats (read this)

1. **The edge may shrink further.** Your original backtest said 64.7% WR; forward reality is
   ~43% (all setups) / ~49% (core 3). The forward numbers are the honest ones — and there is no
   guarantee they persist. Treat +0.5 to +0.6R per trade as a *hope*, not a certainty.
2. **Static 8% is forgiving here because the curve rose early.** A cold opening streak is the
   real risk — that's exactly what the bootstrap stress-tests, and the core-3 plan survives it at
   1% sizing.
3. **Operational compliance:** use bracket orders (SL on immediately), never touch the pending
   order after fill, hold >60s (you will), and check the economic calendar so none of your entry
   times (16:00 / 15:00 / 11:30 GMT) land within ±5 min of high-impact news. Your EOD-close
   trades must be closed manually before the day rollover.
4. This is not financial advice; prop-firm rules can change — re-read the FAQ before funding.

---

## 5. Artifacts

- `strategy_analysis/results/funded_20260908_192246/funded_summary.json` — full edge + funded grid
- `run_funded_sim.py` — the funded-account simulator (rules are constants at the top, easy to tweak)
- `run_funded_native.py` — native-RR + single-setup funded runs
