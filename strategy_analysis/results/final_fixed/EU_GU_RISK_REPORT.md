# EU+GU Risk Sizing Report (Forward Period: 1 Feb – 9 Sep 2026)

> **Setup combination:** EURUSD Thursday (11:30 GMT, RR 1:2, 5-min breakout) +
> GBPUSD Wednesday (15:00 GMT, RR 1:3, 15-min breakout)
>
> **62 trades** over the forward window.

## Risk sizing table (Classic $5,000 account)

| Risk/trade | Return  | Max DD   | 10% Target Day | Consistency |
|------------|---------|----------|----------------|-------------|
| 1.0%       | +17.5%  | 3.77%    | day 51         | ✅ pass     |
| **1.5%**   | **+26.9%** | **5.62%** | **day 32**  | ✅ pass     |
| 2.0%       | +36.7%  | 7.44%    | —              | ✅ pass     |
| 2.15%      | +39.8%  | 8.00%    | —              | ✅ pass     |
| 2.5%       | —       | 9.24%    | BREACHES day 17 | ❌         |

- **2.15%** is the exact ceiling (8.00% max DD = hard breach limit).
- **2.5%** breaches on day 17 (DD 9.24% > 8%).

## Consistency rule (no single day > 25% of total profit)

All risk levels tested pass: best-day share ranges from 18.9% to 20.0%.

## Recommendation

**1.5% risk per trade** on the EU+GU combination:
- +26.9% return over 7 months on a $5K Classic account
- Max drawdown 5.62% (well under 8% hard breach)
- 10% profit target reached by day 32
- Comfortable margin below the 8% ceiling

## Why EU+GU over Core-3

From the forward results:

| Combination     | Forward Return | Max DD  | Trades | Breach Risk |
|-----------------|---------------|---------|--------|-------------|
| **EU+GU**       | **+17.5%**    | **3.77%** | 62   | Very low    |
| Core-3 (EU+GU+UC) | +15.3%     | 6.60%   | 94     | Moderate    |
| EURUSD only     | +13.2%        | 2.05%   | 31     | Very low    |
| GBPUSD only     | +3.8%         | —       | 31     | Low         |
| USDCAD only     | −1.8%         | —       | 32     | N/A (loser) |

USDCAD Monday (netR −1.7) drags down the Core-3 combination. Dropping it improves
both return and risk profile. EURUSD Thursday is the strongest single setup (PF 1.90).

## Notes

- All R-multiples use the corrected convention: `R = pnl_pips / risk_pips` (actual
  result, not full RR on every win).
- Forward period only — in-sample data had known mislabeling issues (EURUSD M5 file
  was actually M15; GBPUSD M15 file was actually M5).
- No EA change has been applied yet — these numbers are from the Python backtest
  layer only.
