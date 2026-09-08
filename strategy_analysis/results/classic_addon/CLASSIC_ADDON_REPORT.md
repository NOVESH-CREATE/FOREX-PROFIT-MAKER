# FundYourFX Classic $5,000 + ADD-ONS — "Bought on 1 Feb 2026" progress report

*Prepared 2026-09-09. Strategy: **core-3 ORB plan** (Mon USDCAD RR3 · Wed GBPUSD RR3 ·
Thu EURUSD RR2), original close-breakout entry, 1 trade/day. Risk = fixed % of the account
level per trade. Based on your actual forward trade log (Feb 2 → Sep 7 2026).*

## Your add-ons (applied in this run)

| Rule | Classic default | **With your add-ons** |
|---|---|---|
| Profit split | 50% day one | **90% from day one** |
| Max loss (static DD) | 6% | **8%** |
| (everything else unchanged) | 10% target · 4% daily DD · 6 min days · fee refund after 2 payouts · scale every 3 payouts | same |

> Assumption I made: I model **90% flat for every payout** (your words). If the standard ladder
> still climbs to 95% after 3 payouts, payouts #4+ are ~5.6% bigger (e.g. $810 → $855).
> The 25% payout rule (best day ≤ 25% of total profit) still applies, so you actually cash at
> ~**12% profit**, not 10%.

## Pass probability (bootstrap, 8% max-loss add-on)

| Risk/trade | P(pass without breaching) |
|---|---|
| 0.5% ($25) | **100%** |
| **1.0% ($50)** | **98.9%** |
| 2.0% ($100) | 92.7% |

---

## 1. If you bought on 1 Feb 2026 — exactly where you'd be TODAY (9 Sep 2026)

**At 1% risk ($50/trade):**

| # | Date | Account level | Gross profit | **You get (90%)** | Note |
|---|---|---|---|---|---|
| 1 | **18 Feb** | $5,000 | $600 | **$540** | first payout |
| 2 | **6 Apr** | $5,000 | $650 | **$585** | ✅ **FEE REFUNDED** |
| 3 | **13 May** | $5,000 | $700 | **$630** | ✅ **SCALE → $7,500** |
| 4 | 9 Jul | $7,500 | $900 | **$810** | |
| 5 | 3 Aug | $7,500 | $900 | **$810** | |
| 6 | **3 Sep** | $7,500 | $900 | **$810** | ✅ **SCALE → $10,000** |

**→ TODAY (9 Sep 2026):**
- **6 payouts completed, total paid out to you = $4,185**
- **Your account has already scaled from $5,000 → $7,500 → $10,000**
- Current balance = **$9,900** (cycle 7 at the $10k level, 1 trade in so far: a −1R loss on 7 Sep)
- **Your account fee was refunded at payout #2** (the exact $ depends on your invoice + add-on price)

**Safer plan — 0.5% risk ($25/trade):** 3 payouts by today = **$1,417.50 paid out**
($450 + $472.50 + $495), scaled once to $7,500, current balance $7,912.50.

---

## 2. Full projection (continuing the same pace, 1% risk)

| # | ~When | Level | Gross | **You get (90%)** | Cumulative |
|---|---|---|---|---|---|
| 7 | Oct 2026 | $10,000 | $1,300 | $1,170 | $5,355 |
| 8 | Nov 2026 | $10,000 | $1,400 | $1,260 | $6,615 |
| 9 | Dec 2026 | $10,000 | $1,300 | $1,170 | $7,785 | SCALE → $25,000 |
| 10 | Feb 2027 | $25,000 | $3,250 | $2,925 | $10,710 |
| 11 | Mar 2027 | $25,000 | $3,000 | $2,700 | $13,410 |
| 12 | Apr 2027 | $25,000 | $3,000 | $2,700 | $16,110 | SCALE → $60,000 |
| 13 | May 2027 | $60,000 | $7,800 | $7,020 | $23,130 |
| 14 | Jun 2027 | $60,000 | $7,800 | $7,020 | $30,150 |
| 15 | Aug 2027 | $60,000 | $8,400 | $7,560 | $37,710 | SCALE → $150,000 |
| 16 | Sep 2027 | $150,000 | $19,500 | $17,550 | $55,260 |
| 17 | Nov 2027 | $150,000 | $19,500 | $17,550 | $72,810 |
| 18 | Dec 2027 | $150,000 | $19,500 | $17,550 | $90,360 | at max ladder |

**Cumulative take-home:** ~$4,185 after 6 payouts · ~$7,785 after 9 · ~$16,110 after 12 ·
~$37,710 after 15 · ~$90,360 after 18 (about 22 months from start).

---

## 3. Equity curve

See `equity_curve.png` next to this file:
- **Blue** = account balance (1% risk, compounds; sawtooth = payout withdrawals reset to level)
- **Red dashed** = 8% static max-loss floor (add-on) — you stay far above it
- **Green dots** = payouts (P1…P6 with the $ you receive)
- **Purple "SCALE"** = capital jumps $7,500 / $10,000 / $25,000…
- **Orange "FEE REFUND"** = at payout #2

Drawdown reality: worst streak = 5 losses in a row = −5% at 1% risk (inside the 8% floor);
worst single day = 1 loss (−1%), far under the 4% daily limit.

---

## 4. Caveats (please read)

1. **Dates after Sep 2026 are a replay** of your real 7-month record, so treat the "~month" column
   as the honest timeline, not the exact dates. Everything up to **9 Sep 2026 is your real data**.
2. The first 3 payouts were fast because **Feb–Mar 2026 was a hot streak** (your log opened
   +3R, +3R, +3R…). Plan on **~5 weeks per payout** as the realistic average, not 3.
3. These are your **out-of-sample forward results (~49% WR on core-3)**, not the inflated 64.7%
   backtest — the edge can still degrade, which is why risk stays at 1% (or 0.5%).
4. **90% flat** is my assumption; confirm whether it climbs to 95% (it only changes payouts #4+).
5. **Operational rules still bind:** SL within 3 min, hold >60s, no entry ±5 min around high-impact
   news (15:45 / 15:00 / 11:30 GMT), max 2 trades/instrument/direction, no martingale/grid/copying.
6. Not financial advice; prop rules can change — re-read the FAQ before funding.
   **Tell me your exact fee ($) and I'll show the fee-refund line in dollars.**

---

## Artifacts

- `strategy_analysis/results/classic_addon/equity_curve.png` — chart
- `strategy_analysis/results/classic_addon/equity_10.csv` / `equity_5.csv` — curve data
- `strategy_analysis/results/classic_addon/schedule_10.json` / `schedule_5.json` — payout schedule
- `run_classic_addon.py` — simulator (rules + add-ons are constants at the top)
