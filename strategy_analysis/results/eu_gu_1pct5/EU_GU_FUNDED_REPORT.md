# FINAL PLAN — EURUSD + GBPUSD only @ 1.5% risk (USDCAD dropped)

> **Prepared 13 Sep 2026 · BACKTEST REBUILT ON THE CURRENT STRATEGY ONLY** · FundYourFX Classic $5,000 + add-ons (90% split, 8% max loss) · risk **1.5% of the account level per trade** · rules corrected from the official FYFX pages: **NO consistency rule, one-time 8% target, minimum $150 payout, fee refund after the 3rd payout, scaling every 3 payouts** · every fee refund buys a NEW $5K account.

## Why this report supersedes every older backtest number

The older in-sample reports in this repo were produced when the strategy was still the **all-assets scenario list** (Mon USDCAD, Wed GBP, Thu USDCAD+EUR+GBP, Fri USDCAD) — and part of that old pre-Feb data was later found mislabeled and removed. The strategy is now **EU+GU only**, so the backtest was **rebuilt from scratch on the current two legs only**, using the same engine and the gate-verified genuine data. Where a leg cannot be backtested, this report says so explicitly instead of filling the gap with another asset.

## The plan

| Day | Pair | Mother candle (GMT) | Entry | RR |
|---|---|---|---|---|
| Wednesday | GBPUSD | 15:00 (15-min) | 15-min close-breakout | 1:3 |
| Thursday | EURUSD | 11:30 (15-min) | **5-min close-breakout** | 1:2 |

~~Monday USDCAD~~ — **dropped**: 32 forward Mondays at the current settings made netR −1.68R (PF 0.88); every tested variant (5-min entry, RR 1:2) was worse. See `../m5_breakout/M5_BREAKOUT_REPORT.md`.

## ✅ GU backtest leg INCLUDED (uploaded GBPUSD file validated)

Source: `GBPUSD_M15_202509011745_202609112045.csv [M15 CSV (parsed)]`. Before use it passed **GATE D** (candle-for-candle match vs the real M15 .htm export on the overlap) and **GATE E** (forward-window trades identical to the real-data trades) — no mislabeled data can enter silently.

## ✅ Verification gates

| Gate | Check | Result |
|---|---|---|
| A | forward EU+GU sequence reproduces `EU_GU_RISK_REPORT.md` (+17.50% @1% / maxDD 3.77%; +26.90% @1.5% / 5.62%) | PASS |
| B | forward counts: 31 EUR + 31 GBP | PASS |
| C | full-year EU sequence (M5-rebuilt) ≡ standard forward EU sequence on the overlap | PASS |

## Trade results — backtest rebuilt on EU+GU legs only

### BACKTEST window — 1 Sep 2025 → 31 Jan 2026 (EU + GU)

- EURUSD Thu 11:30 RR2 (5-min): **20 trades** · 11W/9L · WR 55.0% · netR +13.00R · PF 2.44 · monthly: 2025-09: +2.00R (4t), 2025-10: +7.00R (5t), 2025-11: +2.00R (4t), 2025-12: -3.00R (3t), 2026-01: +5.00R (4t)
- GBPUSD Wed 15:00 RR3 (15-min): **22 trades** · 14W/8L · WR 63.6% · netR +7.66R · PF 2.91 · monthly: 2025-09: +3.42R (4t), 2025-10: -1.33R (5t), 2025-11: +2.18R (4t), 2025-12: +0.58R (5t), 2026-01: +2.81R (4t)
- **BACKTEST COMBINED: 42 trades · 25W/17L · WR 59.5% · netR +20.66R · PF 2.59**
- Combined monthly: 2025-09: +5.42R (8t), 2025-10: +5.67R (10t), 2025-11: +4.18R (8t), 2025-12: -2.42R (8t), 2026-01: +7.81R (8t)
- vs the forward period (+16.63R): the EU(+GU) strategy is **two-window stable** — no in-sample/out-sample flip.

### FORWARD window — 1 Feb → 8 Sep 2026 (EU+GU, the live-verified leg)

| Setup | Trades | W/L | WR | netR |
|---|---|---|---|---|
| GBPUSD Wed 15:00 RR3 (15-min) | 31 | 15/16 | 48.4% | +3.93R |
| EURUSD Thu 11:30 RR2 (5-min) | 31 | 16/15 | 51.6% | +12.70R |
| **EU+GU combined** | **62** | **31/31** | **50.0%** | **+16.63R** |

Monthly: 2026-02: +4.49R (8t), 2026-03: -1.20R (8t), 2026-04: +0.90R (10t), 2026-05: +3.60R (8t), 2026-06: +4.63R (8t), 2026-07: +0.89R (10t), 2026-08: +0.55R (8t), 2026-09: +2.77R (2t)

### FULL-YEAR sequence — 1 Sep 2025 → 8 Sep 2026 (EU full-year + GU from Feb)

- **82 trades** (51 EUR + 31 GBP) · netR **+29.63R**

## 💰 FUNDED-ACCOUNT RESULTS — three windows, all @ 1.5% risk, corrected FYFX rules

| Window | Trades | Payouts | Accounts | Paid out | Combined balance | TOTAL PROFIT |
|---|---|---|---|---|---|---|
| BACKTEST Sep'25–Jan'26 (EU + GU) | 42 | 6 | 2 | $1,806.59 | $13,020.82 | **$2,327.41** |
| FORWARD Feb–Sep'26 (EU+GU) — **the as-of-today reality** | 62 | 5 | 2 | $1,188.11 | $13,039.68 | **$1,727.79** |
| FULL YEAR Sep'25–Sep'26 (EU + GU full) | 104 | 24 | 4 | $9,411.08 | $82,650.00 | **$9,561.08** |

**Read it like this:** the FORWARD row is what actually happened to the plan-of-record account (bought 1 Feb 2026) — that is your real position as of 9 Sep 2026. The BACKTEST and FULL-YEAR rows answer "what would the account have done if this exact strategy + these rules had run from Sep 2025" — they are the honest maximum-history view, limited to the data that exists.

### As-of-today ledger (FORWARD window — your real account)

| Date | Account | Payout # | Gross | You receive (90%) | Cumulative | Note |
|---|---|---|---|---|---|---|
| 2026-02-25 | #1 | 1 | $412.05 | **$370.84** | $370.84 | — |
| 2026-05-28 | #1 | 2 | $172.47 | **$155.22** | $526.06 | — |
| 2026-06-18 | #1 | 3 | $204.91 | **$184.42** | $710.48 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |
| 2026-07-15 | #1 | 4 | $325.57 | **$293.01** | $1,003.49 | — |
| 2026-08-06 | #1 | 5 | $205.14 | **$184.62** | $1,188.11 | — |

### Full-year hypothetical cascade ledger (what the data allows)

| Date | Account | Payout # | Gross | You receive (90%) | Cumulative | Note |
|---|---|---|---|---|---|---|
| 2025-09-25 | #1 | 1 | $406.18 | **$365.56** | $365.56 | — |
| 2025-10-16 | #1 | 2 | $275.43 | **$247.89** | $613.45 | — |
| 2025-11-06 | #1 | 3 | $315.62 | **$284.06** | $897.51 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |
| 2025-11-27 | #1 | 4 | $221.93 | **$199.74** | $1,097.25 | — |
| 2026-01-14 | #2 | 1 | $404.03 | **$363.63** | $1,460.88 | — |
| 2026-01-15 | #1 | 5 | $384.12 | **$345.71** | $1,806.59 | — |
| 2026-02-04 | #2 | 2 | $383.95 | **$345.56** | $2,152.15 | — |
| 2026-02-05 | #1 | 6 | $238.43 | **$214.59** | $2,366.74 | **SCALE → $10,000** |
| 2026-02-25 | #2 | 3 | $326.42 | **$293.78** | $2,660.52 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |
| 2026-02-26 | #1 | 7 | $652.84 | **$587.56** | $3,248.08 | — |
| 2026-04-08 | #1 | 8 | $184.08 | **$165.67** | $3,413.75 | — |
| 2026-05-28 | #1 | 9 | $310.86 | **$279.77** | $3,693.52 | **SCALE → $25,000** |
| 2026-05-28 | #2 | 4 | $258.70 | **$232.83** | $3,926.35 | — |
| 2026-06-10 | #3 | 1 | $475.04 | **$427.54** | $4,353.89 | — |
| 2026-06-18 | #1 | 10 | $1,024.55 | **$922.10** | $5,275.99 | — |
| 2026-06-18 | #2 | 5 | $307.37 | **$276.63** | $5,552.62 | — |
| 2026-07-15 | #1 | 11 | $1,085.23 | **$976.71** | $6,529.33 | — |
| 2026-07-15 | #2 | 6 | $325.57 | **$293.01** | $6,822.34 | **SCALE → $10,000** |
| 2026-07-15 | #3 | 2 | $217.05 | **$195.34** | $7,017.68 | — |
| 2026-08-06 | #1 | 12 | $683.79 | **$615.42** | $7,633.10 | **SCALE → $60,000** |
| 2026-08-06 | #2 | 7 | $273.52 | **$246.17** | $7,879.27 | — |
| 2026-09-03 | #1 | 13 | $1,252.20 | **$1,126.98** | $9,006.25 | — |
| 2026-09-03 | #2 | 8 | $208.70 | **$187.83** | $9,194.08 | — |
| 2026-09-03 | #3 | 3 | $241.11 | **$217.00** | $9,411.08 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |

### Rule sensitivity (FORWARD window, 1.5% risk)

| Rule variant | Payouts by 9 Sep | Accounts | Total paid out | Combined balance |
|---|---|---|---|---|
| **PLAN: 8% target once · refund @ payout 3 (official)** | 5 | 2 | $1,188.11 | $13,039.68 |
| **refund @ payout 2 (your earlier reading)** | 7 | 3 | $1,825.95 | $18,015.23 |
| **10% target (official Classic) · refund @ payout 3** | 3 | 2 | $1,028.90 | $12,910.88 |

*PLAN row = your corrected reading (8% target once + official refund after payout #3). If your dashboard shows 10% (official Classic) or a payout-#2 refund, use those rows.*

### What happens next (mechanics, not prophecy)

- After the target is met, every time a cycle's profit reaches **$167+ (= $150 received at 90%)** and 6 trading days have passed, the payout fires — at the current +2.3R/month pace that is roughly **every 3–5 weeks early on**, faster once two accounts run.
- **Payout #3 = fee refund = account #2** (already inside the forward window: opened 18 Jun 2026).
- Re-run `python3 strategy_analysis/eu_gu_funded_plan.py` whenever you export fresh data — and upload GBPUSD M5 full-year to complete the GU backtest leg.

## 1.5% vs 1.0% (FORWARD window)

| Metric | 1.0% risk | **1.5% risk (PLAN)** |
|---|---|---|
| Total paid out by 9 Sep | $748.55 | **$1,188.11** |
| Combined balance | $12,600.00 | **$13,039.68** |
| Combined funded | $12,500 | **$12,500** |
| Pure-compound return (no payout resets) | +17.50% | **+26.90%** |
| Pure-compound max drawdown | 3.77% | **5.62%** (vs 8% breach: safe) |
| Worst-case consecutive-loss breach? | no | **no** (needs 5.3 straight full losses; worst seen = 3) |

### Safety record of the as-of-today run @ 1.5% (verified)

- Balance **never closed below -4.80% vs the funded level** on any trade close — the 8% max-loss line was never remotely threatened.
- Worst single DAY: −1.00R = **−1.50%** of level vs the 4% daily-DD limit (one setup trades per day, so a day can lose at most −1.5%).
- Max consecutive losses: **3**; no breach, no daily-DD event in any account, in any window.

## Files & rerun

- `eu_gu_funded_equity.html` / `index.html` — interactive 3-panel chart (full-year cascade: equity vs max-loss floor · cumulative payouts · active accounts)
- `eu_gu_summary.json` — machine-readable summary (all windows + sensitivity + payout ledgers)
- `trades_EU_GBP.csv` (forward), `trades_EU_backtest.csv`, `trades_GU_backtest.csv`, `trades_FULLYEAR_EU_GU.csv` — full trade lists
- Rerun: `python3 strategy_analysis/eu_gu_funded_plan.py` — gates re-verify everything on every run.

