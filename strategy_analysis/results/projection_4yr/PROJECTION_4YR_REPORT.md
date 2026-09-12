# 4-Year Projection — EU+GU FYFX plan (Monte Carlo, not a prediction)

> **Method (what funds actually do):** simulate thousands of possible 4-year futures by resampling the 104 gate-verified trades (block bootstrap + a fitted 2-state Markov regime model), run every future through the *real* FYFX cascade (8% target once, $150 payouts, 90% split, refund→new account, scale every 3 payouts, 8% breach → rebuy fresh $5K), and read the DISTRIBUTION. The future is a range, never a line.

## Diagnostics on the verified history

- 104 trades · mean **+0.359R** · sd 1.249R · t = 2.93
- ADF stationarity p = 0.0000 → no statistical evidence of edge decay so far
- 2-state regime fit: mixed days (-1.00R) vs weak days (+1.05R), persistence 0.26/0.62 — streaky, exactly what the block bootstrap preserves.

## The 4-year distribution (total profit = cash payouts + in-cycle)

| Scenario | Median 4y profit | p5 (bad luck) | p95 (good luck) | P(end < 0) | P(no breach) | P(funded ≥ $25K) | P(funded ≥ $60K) |
|---|---|---|---|---|---|---|---|
| A. FULL EDGE (block bootstrap) | **$189,912** | $71,441 | $261,422 | 0.0% | 60.2% | 100.0% | 100.0% |
| B. EDGE DECAY 50% | **$12,131** | $-15 | $103,584 | 5.1% | 3.6% | 99.9% | 82.4% |
| C. EDGE DECAY 75% DOWN | **$1,566** | $-1,526 | $25,927 | 16.6% | 0.1% | 99.7% | 39.6% |
| D. NO EDGE (survival test) | **$-622** | $-2,925 | $1,644 | 69.2% | 0.0% | 99.6% | 17.2% |
| E. FULL EDGE + 0.5 pip COST/trade | **$142,416** | $36,742 | $232,909 | 0.2% | 42.0% | 100.0% | 100.0% |
| F. REGIME-MODEL future (Markov) | **$143,736** | $27,869 | $256,970 | 0.0% | 27.8% | 100.0% | 100.0% |

## How to read this

1. **Full edge continues (A):** median **$189,912** in 4 years; even the unlucky 5% path makes $71,441. Chance you're at a $60K level at some point: 100%.
2. **Edge decays 50% (B):** still $12,131 median — the multi-account machinery is forgiving because payouts skim profits continuously instead of letting one big cycle ride.
3. **No edge at all (D):** median $-622 — this is what pure variance + fees look like. This row is the honest reason funds demand an edge premium before scaling.
4. **Costs matter (E):** just 0.5 pip/trade of slippage+spread shifts the median by ~$47,496 over 4 years. Log your real fills.
5. **Breaches:** 0.50 average account deaths over 4 years in the full-edge world — the cascade design (small accounts, refund-bought replacements) treats a breach as a business expense, not a catastrophe.

## What this CANNOT tell you (be honest with yourself)

- It resamples the PAST year's trade distribution. A genuine regime change (the edge disappearing) is only covered by scenarios B-D, not by the data.
- 104 trades is a small sample; the confidence bands are wide. Every new month of live trades tightens them — re-run monthly.
- FYFX counterparty/rules risk (the firm changing terms or closing) is not simulatable — that risk dwarfs everything above at a 4-year horizon.

## Files

- `projection_4yr_fan.html` — fan chart of the profit distribution
- `projection_summary.json` — full numbers per scenario
- Rerun: `python3 strategy_analysis/project_4yr.py` (re-draws with a fixed seed; swap in updated trade data monthly)

