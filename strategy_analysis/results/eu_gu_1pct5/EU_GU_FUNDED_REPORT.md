# FINAL PLAN — EURUSD + GBPUSD only @ 1.5% risk (USDCAD dropped)

> **Prepared 13 Sep 2026 · window 1 Feb → 9 Sep 2026** · FundYourFX Classic $5,000 + add-ons (90% split, 8% max loss) · risk **1.5% of the account level per trade** · **rules corrected from the official FYFX pages: NO consistency rule, one-time 8% target, minimum $150 payout, fee refund after the 3rd payout, scaling every 3 payouts** · every fee refund buys a NEW $5K account.

## The plan

| Day | Pair | Mother candle (GMT) | Entry | RR |
|---|---|---|---|---|
| Wednesday | GBPUSD | 15:00 (15-min) | 15-min close-breakout | 1:3 |
| Thursday | EURUSD | 11:30 (15-min) | **5-min close-breakout** | 1:2 |

~~Monday USDCAD~~ — **dropped**: 32 forward Mondays at the current settings made netR −1.68R (PF 0.88); every tested variant (5-min entry, RR 1:2) was worse. See `../m5_breakout/M5_BREAKOUT_REPORT.md`.

## ✅ Rules corrected from the official FYFX pages (this supersedes the old 25% model)

You were right — checking fundyourfx.io + the official helpdesk:

- **NO consistency rule** — the FYFX homepage advertises "No Consistency Rule" on funded accounts. The old 25%-best-day condition in the previous model is **removed**.
- **One-time target:** make **8%** once → payouts unlocked (official Instant Funding Pro target is 8%; Classic lists 10% — modelled at 8% per your plan; one line in the script flips it if your dashboard says 10%).
- **Minimum payout $150** (what you receive) once the target is met — confirmed by the official helpdesk FAQ.
- **Fee refund after the 3rd successful payout** (official) → buys the NEW $5K account. *(The old model assumed refund after payout #2 — the sensitivity table below shows both.)*
- **Scaling every 3 payouts**: $5K → $7.5K → $10K → $25K → $60K → $150K (confirmed).
- Kept from your add-ons: **90% split from day one**, **8% static max loss**, 4% daily DD (soft), 6 trading days minimum between payouts.

## ✅ Verification gate — sequence identical to the published EU+GU report

Before simulating, this exact trade sequence was checked against `../../final_fixed/EU_GU_RISK_REPORT.md`:

| Risk | This run | Published report | Match |
|---|---|---|---|
| 1.0% | +17.50% / maxDD 3.77% | +17.5% / 3.77% | ✅ |
| 1.5% | +26.90% / maxDD 5.62% | +26.9% / 5.62% | ✅ |

62 trades (31 EUR + 31 GBP), netR **+16.63R** — the same signal sequence behind every repo number. No data or engine drift.

## Trade results (1 Feb → 8 Sep 2026)

| Setup | Trades | W/L | WR | netR | Note |
|---|---|---|---|---|---|
| GBPUSD Wed 15:00 RR3 (15-min) | 31 | 15/16 | 48.4% | +3.93R | steady, PF ~1.5 |
| EURUSD Thu 11:30 RR2 (5-min) | 31 | 16/15 | 51.6% | +12.70R | the powerhouse: +13.0R backtest / +12.7R forward (verified) |
| **EU+GU combined** | **62** | **31/31** | **50.0%** | **+16.63R** | **+26.90% compounded @1.5%** |

Monthly netR:

- 2026-02: +4.49R (8t), 2026-03: -1.20R (8t), 2026-04: +0.90R (10t), 2026-05: +3.60R (8t), 2026-06: +4.63R (8t), 2026-07: +0.89R (10t), 2026-08: +0.55R (8t), 2026-09: +2.77R (2t)

7 of 8 months positive (September is a 2-trade partial month). The only negative month is March (−1.20R) — well inside the plan's 5.62% max drawdown envelope.

## 💰 THE ANSWER — funded-account results @ 1.5% risk, CORRECTED FYFX rules (as of 9 Sep 2026)

- **Total paid out to you (cash, 90% split): $1,188.11**
- **Active accounts: 2**  (account #1 + 1 bought with fee refunds)
- **Combined funded capital: $12,500**
- **Combined account balance: $13,039.68** (= funded + $539.68 current-cycle profit)
- **TOTAL PROFIT MADE (payouts + in-account cycle profit): $1,727.79**

### Payout ledger

| Date | Account | Payout # | Gross | You receive (90%) | Cumulative | Note |
|---|---|---|---|---|---|---|
| 2026-02-25 | #1 | 1 | $412.05 | **$370.84** | $370.84 | — |
| 2026-05-28 | #1 | 2 | $172.47 | **$155.22** | $526.06 | — |
| 2026-06-18 | #1 | 3 | $204.91 | **$184.42** | $710.48 | **FEE REFUND → new $5K account**, **SCALE → $7,500** |
| 2026-07-15 | #1 | 4 | $325.57 | **$293.01** | $1,003.49 | — |
| 2026-08-06 | #1 | 5 | $205.14 | **$184.62** | $1,188.11 | — |

### Account-by-account

| Account | Opened | Trades taken | Payouts | Paid out | Level now | Balance now | Status |
|---|---|---|---|---|---|---|---|
| #1 | 2026-02-01 | 62 | 5 | $1,188.11 | $7,500 | $7,656.53 | healthy |
| #2 | 2026-06-18 | 23 | 0 | $0.00 | $5,000 | $5,383.16 | healthy |

### Rule sensitivity (same 62 trades, different rule readings)

| Rule variant | Payouts by 9 Sep | Accounts | Total paid out | Combined balance |
|---|---|---|---|---|
| **PLAN: 8% target once · refund @ payout 3 (official)** | 5 | 2 | $1,188.11 | $13,039.68 |
| **refund @ payout 2 (your earlier reading)** | 7 | 3 | $1,825.95 | $18,015.23 |
| **10% target (official Classic) · refund @ payout 3** | 3 | 2 | $1,028.90 | $12,910.88 |

*If your dashboard shows a 10% target (official Classic) use the last row; if your fee refund lands at payout #2 (your earlier reading) use the second row. The PLAN row follows your corrected reading: 8% target once + official refund after payout #3.*

### What happens next (mechanics, not prophecy)

- After the target is met, every time a cycle's profit reaches **$167+ (= $150 received at 90%)** and 6 trading days have passed, the payout fires — at the current +2.3R/month pace that is roughly **every 3–5 weeks early on**, faster once two accounts run.
- **Payout #3 = fee refund = account #2** (already inside the window above).
- Re-run `python3 strategy_analysis/eu_gu_funded_plan.py` whenever you export fresh data — the ledger, cascade and chart update automatically.

## 1.5% vs 1.0% — why 1.5% is the right call

| Metric | 1.0% risk | **1.5% risk (PLAN)** |
|---|---|---|
| Total paid out by 9 Sep | $748.55 | **$1,188.11** |
| Combined balance | $10,310.37 | **$13,039.68** |
| Combined funded | $10,000 | **$12,500** |
| Pure-compound return (no payout resets) | +17.50% | **+26.90%** |
| Pure-compound max drawdown | 3.77% | **5.62%** (vs 8% breach: safe) |
| Worst-case consecutive-loss breach? | no | **no** (needs 5.3 straight full losses; worst seen = 3) |

5.62% max DD against an 8% hard breach leaves a 2.4% buffer — the largest return that still keeps the account comfortably alive. 2.0% (7.44% DD) and 2.15% (8.00% DD) leave no margin; 2.5% breaches.

### Safety record of this exact 7-month run @ 1.5% (verified)

- Balance **never closed below -4.80% vs the funded level** on any trade close — the 8% max-loss line was never remotely threatened.
- Worst single DAY: −1.00R = **−1.50%** of level vs the 4% daily-DD limit — never close to a daily breach (one setup trades per day, so a day can lose at most −1.5%).
- Max consecutive losses: **3** (a breach would need 5.3 straight full losses at 1.5%).
- No breach, no daily-DD event across every account in the simulation.

## Files & rerun

- `eu_gu_funded_equity.html` / `index.html` — interactive 3-panel chart (equity vs max-loss floor · cumulative payouts · active accounts)
- `eu_gu_summary.json` — machine-readable summary + payout ledger
- `combined_equity.csv`, `account_<n>_equity.csv` — equity series
- `trades_EU_GBP.csv` — the 62-trade signal list
- Rerun: `python3 strategy_analysis/eu_gu_funded_plan.py` — the verification gate re-checks the sequence against the published report on every run and refuses to output if anything drifted.

