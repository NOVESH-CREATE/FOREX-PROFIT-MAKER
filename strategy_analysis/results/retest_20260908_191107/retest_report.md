# ORB Scenario-3: limit-retest entry vs original close-breakout entry

## Entry rule change (everything else identical)
1. Mother (ORB) candle = 15-min candle at setup time.
2. First candle that **closes** beyond the range sets direction (same break rule as before).
3. No entry on that candle. After it closes, place a limit order:
   - close above mother high -> **BUY LIMIT at mother high**
   - close below mother low  -> **SELL LIMIT at mother low**
4. Order fills only if price retraces to the level same day; otherwise it expires at EOD (no trade).
5. SL/TP unchanged: SL = opposite side of mother candle +/- 2 pips buffer, TP = risk x RR.
6. On the fill candle SL is checked before TP; afterwards the original engine takes over.

## Headline

| Metric | In-sample: orig entry | In-sample: retest entry | Forward: orig entry | Forward: retest entry |
|---|---|---|---|---|
| Trades | 136 | 106 | 185 | 168 |
| Wins | 88 | 62 | 79 | 72 |
| Losses | 48 | 44 | 106 | 96 |
| Win rate % | 64.71 | 58.49 | 42.7 | 42.86 |
| Net R | 168.5 | 109.5 | 87.5 | 79.5 |
| Gross profit R | 216.5 | 153.5 | 193.5 | 175.5 |
| Gross loss R | 48.0 | 44.0 | 106.0 | 96.0 |
| Profit factor | 4.51 | 3.49 | 1.83 | 1.83 |
| Avg RR | 2.48 | 2.49 | 2.42 | 2.42 |
| Total pips | 992.2 | 472.9 | 77.5 | 222.0 |
| Max cons. wins | 9 | 6 | 5 | 5 |
| Max cons. losses | 4 | 4 | 9 | 6 |

## Per-setup — in-sample (Aug 2025 - Jan 2026)

| Setup | Orig n | Orig WR% | Orig R | Retest n | Retest WR% | Retest R |
|---|---|---|---|---|---|---|
| Friday | USDCAD | 09:30 PM | 27 | 63.0 | 32.5 | 19 | 47.4 | 12.5 |
| Monday | USDCAD | 09:15 PM | 26 | 61.5 | 38.0 | 22 | 63.6 | 34.0 |
| Thursday | EURUSD | 05:00 PM | 24 | 66.7 | 24.0 | 18 | 61.1 | 15.0 |
| Thursday | GBPUSD | 09:15 AM | 11 | 63.6 | 10.0 | 10 | 70.0 | 11.0 |
| Thursday | USDCAD | 04:45 PM | 22 | 72.7 | 26.0 | 17 | 58.8 | 13.0 |
| Wednesday | GBPUSD | 08:30 PM | 26 | 61.5 | 38.0 | 20 | 55.0 | 24.0 |

## Per-setup — forward (Feb - Sep 2026)

| Setup | Orig n | Orig WR% | Orig R | Retest n | Retest WR% | Retest R |
|---|---|---|---|---|---|---|
| Friday | USDCAD | 09:30 PM | 30 | 36.7 | 8.5 | 27 | 33.3 | 4.5 |
| Monday | USDCAD | 09:15 PM | 32 | 46.9 | 28.0 | 28 | 39.3 | 16.0 |
| Thursday | EURUSD | 05:00 PM | 31 | 51.6 | 17.0 | 27 | 51.9 | 15.0 |
| Thursday | GBPUSD | 09:15 AM | 30 | 30.0 | -3.0 | 29 | 31.0 | -2.0 |
| Thursday | USDCAD | 04:45 PM | 31 | 41.9 | 8.0 | 28 | 46.4 | 11.0 |
| Wednesday | GBPUSD | 08:30 PM | 31 | 48.4 | 29.0 | 29 | 55.2 | 35.0 |