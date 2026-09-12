# FINAL PLAN — EURUSD + GBPUSD only @ 1.5% risk (USDCAD dropped)

> **Prepared 12 Sep 2026 · window 1 Feb → 9 Sep 2026** · FundYourFX Classic $5,000 + add-ons (90% split, 8% max loss) · risk **1.5% of the account level per trade** (the repo-recommended level from `EU_GU_RISK_REPORT.md`) · every fee refund (after an account's 2nd payout) buys a NEW $5K account · scaling every 3 payouts: $5K → $7.5K → $10K → $25K → $60K → $150K.

## The plan

| Day | Pair | Mother candle (GMT) | Entry | RR |
|---|---|---|---|---|
| Wednesday | GBPUSD | 15:00 (15-min) | 15-min close-breakout | 1:3 |
| Thursday | EURUSD | 11:30 (15-min) | **5-min close-breakout** | 1:2 |

~~Monday USDCAD~~ — **dropped**: 32 forward Mondays at the current settings made netR −1.68R (PF 0.88); every tested variant (5-min entry, RR 1:2) was worse. See `../m5_breakout/M5_BREAKOUT_REPORT.md`.

## ✅ Verification gate — sequence identical to the published EU+GU report

Before simulating, this exact trade sequence was checked against `../../final_fixed/EU_GU_RISK_REPORT.md` (published earlier in this repo):

| Risk | This run | Published report | Match |
|---|---|---|---|
| 1.0% | +17.50% / maxDD 3.77% | +17.5% / 3.77% | ✅ |
| 1.5% | +26.90% / maxDD 5.62% | +26.9% / 5.62% | ✅ |

62 trades (31 EUR + 31 GBP), netR **+16.63R** — byte-for-byte the same signal sequence behind every repo number. No data or engine drift.

## Trade results (1 Feb → 8 Sep 2026)

| Setup | Trades | W/L | WR | netR | Note |
|---|---|---|---|---|---|
| GBPUSD Wed 15:00 RR3 (15-min) | 31 | 15/16 | 48.4% | +3.93R | steady, PF ~1.5 |
| EURUSD Thu 11:30 RR2 (5-min) | 31 | 16/15 | 51.6% | +12.70R | the powerhouse: +13.0R backtest / +12.7R forward (verified) |
| **EU+GU combined** | **62** | **31/31** | **50.0%** | **+16.63R** | **+26.90% compounded @1.5%** |

Monthly netR:

- 2026-02: +4.49R (8 trades)
- 2026-03: -1.20R (8 trades) ⚠️
- 2026-04: +0.90R (10 trades)
- 2026-05: +3.60R (8 trades)
- 2026-06: +4.63R (8 trades)
- 2026-07: +0.89R (10 trades)
- 2026-08: +0.55R (8 trades)
- 2026-09: +2.77R (2 trades)

7 of 8 months positive (September is a 2-trade partial month). The only negative month is March (−1.20R) — well inside the plan's 5.62% max drawdown envelope.

## 💰 THE ANSWER — funded-account results @ 1.5% risk (as of 9 Sep 2026)

- **Total paid out to you (cash, 90% split): $593.56**
- **Active accounts: 1**
- **Combined funded capital: $5,000**
- **Combined account balance: $5,588.07** (= funded + $588.07 current-cycle profit)
- **Total profit made (payouts + in-account cycle profit): $1,181.63**

### Payout ledger

| Date | Account | Payout # | Gross | You receive (90%) | Cumulative | Note |
|---|---|---|---|---|---|---|
| 2026-06-04 | #1 | 1 | $659.52 | **$593.56** | $593.56 | — |

### Account-by-account

| Account | Opened | Trades taken | Payouts | Paid out | Level now | Balance now | Status |
|---|---|---|---|---|---|---|---|
| #1 | 2026-02-01 | 62 | 1 | $593.56 | $5,000 | $5,588.07 | healthy |

### Run-rate projection (clearly labelled: NOT backtested — beyond 9 Sep)

- The plan is running at **+2.3R/month** across the two setups (~$171/month at the current level).
- **Why only one payout so far:** cycle 2 is at +$588.07 (11.8% — above the 10% target, past the 6-day minimum), but FYFX's **25% consistency rule** blocks it: the best single day made +3.00R ($225), so the cycle needs ≥ 4 × $225 = **$900** before payout #2 releases — about **+4.2R more ≈ mid-October** at the current pace.
- **Payout #2 = the fee refund = account #2.** The moment that second payout lands, the fee refund buys a NEW $5K Classic account and the compounding cascade starts; every 3rd payout scales the older account up the $5K → $7.5K → $10K ladder. From then on the plan runs two accounts on the same two signals.

## 1.5% vs 1.0% — why 1.5% is the right call

| Metric | 1.0% risk | **1.5% risk (PLAN)** |
|---|---|---|
| Total paid out by 9 Sep | $473.66 | **$593.56** |
| Combined balance | $5,305.44 | **$5,588.07** |
| Combined funded | $5,000 | **$5,000** |
| Pure-compound return (no payout resets) | +17.50% | **+26.90%** |
| Pure-compound max drawdown | 3.77% | **5.62%** (vs 8% breach: safe) |
| Worst-case consecutive-loss breach? | no | **no** (needs 5.3 straight full losses; worst seen = 3) |

5.62% max DD against an 8% hard breach leaves a 2.4% buffer — the largest return that still keeps the account comfortably alive. 2.0% (7.44% DD) and 2.15% (8.00% DD) leave no margin; 2.5% breaches.

### Safety record of this exact 7-month run @ 1.5% (verified)

- Balance **never closed below the funded level** on any trade close — the 8% max-loss line was never remotely threatened.
- Worst single DAY: −1.00R = **−1.50%** of level vs the 4% daily-DD limit — never close to a daily breach (one setup trades per day, so a day can lose at most −1.5%).
- Max consecutive losses: **3** (a breach would need 5.3 straight full losses at 1.5%).
- No breach, no daily-DD event, account healthy as of 9 Sep 2026.

## Files & rerun

- `eu_gu_funded_equity.html` / `index.html` — interactive 3-panel chart (equity vs max-loss floor · cumulative payouts · active accounts)
- `eu_gu_summary.json` — machine-readable summary + payout ledger
- `combined_equity.csv`, `account_<n>_equity.csv` — equity series
- `trades_EU_GBP.csv` — the 62-trade signal list
- Rerun: `python3 strategy_analysis/eu_gu_funded_plan.py` — the verification gate re-checks the sequence against the published report on every run and refuses to output if anything drifted.

