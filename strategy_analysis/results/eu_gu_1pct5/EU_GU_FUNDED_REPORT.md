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

## ⚠️ One honest data limitation: the GU backtest leg

The repo's only GBPUSD file starts **1 Feb 2026** — there is **no GBPUSD data before the forward window**, so the Wednesday GBPUSD leg **cannot be backtested yet**. (EURUSD can: your full-year M5 export covers Sep 2025 → Sep 2026.) **→ Upload `GBPUSD_M5` full-year (Sep 2025 → Sep 2026) and re-run this script; the GU backtest leg completes automatically.** Until then, the backtest window is **EURUSD-only**, and this is stated on every number below.

## ✅ Verification gates

| Gate | Check | Result |
|---|---|---|
| A | forward EU+GU sequence reproduces `EU_GU_RISK_REPORT.md` (+17.50% @1% / maxDD 3.77%; +26.90% @1.5% / 5.62%) | PASS |
| B | forward counts: 31 EUR + 31 GBP | PASS |
| C | full-year EU sequence (M5-rebuilt) ≡ standard forward EU sequence on the overlap | PASS |

## Trade results — backtest rebuilt on EU+GU legs only

### BACKTEST window — 1 Sep 2025 → 31 Jan 2026 (**EURUSD only** — no GBP data exists pre-Feb)

- **20 trades** (20 of 22 Thursdays; 25 Dec + 1 Jan holidays) · **11W/9L · WR 55.0%** · netR **+13.00R** · PF 2.44
- Monthly: 2025-09: +2.00R (4t), 2025-10: +7.00R (5t), 2025-11: +2.00R (4t), 2025-12: -3.00R (3t), 2026-01: +5.00R (4t)
- vs its own forward period (+12.70R forward): the EUR leg is **two-window stable** — no in-sample/out-sample flip.

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
| BACKTEST Sep'25–Jan'26 (EU only — GU data missing) | 20 | 3 | 2 | $877.50 | $10,150.00 | **$1,027.50** |
| FORWARD Feb–Sep'26 (EU+GU) — **the as-of-today reality** | 62 | 5 | 2 | $1,188.11 | $13,039.68 | **$1,727.79** |
| FULL YEAR Sep'25–Sep'26 (EU full + GU from Feb) | 82 | 13 | 3 | $4,103.27 | $38,432.63 | **$5,035.90** |

**Read it like this:** the FORWARD row is what actually happened to the plan-of-record account (bought 1 Feb 2026) — that is your real position as of 9 Sep 2026. The BACKTEST and FULL-YEAR rows answer "what would the account have done if this exact strategy + these rules had run from Sep 2025" — they are the honest maximum-history view, limited to the data that exists (GU leg missing pre-Feb).

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
| 2025-10-23 | #1 | 1 | $525.00 | **$472.50** | $472.50 | — |
| 2025-12-04 | #1 | 2 | $225.00 | **$202.50** | $675.00 | — |
| 2026-01-29 | #1 | 3 | $225.00 | **$202.50** | $877.50 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |
| 2026-02-19 | #1 | 4 | $471.57 | **$424.42** | $1,301.92 | — |
| 2026-03-12 | #1 | 5 | $216.65 | **$194.99** | $1,496.91 | — |
| 2026-05-21 | #2 | 1 | $434.52 | **$391.06** | $1,887.97 | — |
| 2026-05-28 | #1 | 6 | $188.55 | **$169.69** | $2,057.66 | **SCALE → $10,000** |
| 2026-06-11 | #2 | 2 | $429.91 | **$386.92** | $2,444.58 | — |
| 2026-06-18 | #1 | 7 | $409.82 | **$368.84** | $2,813.42 | — |
| 2026-07-15 | #1 | 8 | $434.09 | **$390.68** | $3,204.10 | — |
| 2026-08-06 | #1 | 9 | $273.52 | **$246.17** | $3,450.27 | **SCALE → $25,000** |
| 2026-08-06 | #2 | 3 | $203.81 | **$183.42** | $3,633.69 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |
| 2026-09-03 | #1 | 10 | $521.75 | **$469.58** | $4,103.27 | — |

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
| Combined balance | $10,310.37 | **$13,039.68** |
| Combined funded | $10,000 | **$12,500** |
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
- `trades_EU_GBP.csv` (forward), `trades_EU_backtest.csv`, `trades_FULLYEAR_EU_GU.csv` — full trade lists
- Rerun: `python3 strategy_analysis/eu_gu_funded_plan.py` — gates re-verify everything on every run.

