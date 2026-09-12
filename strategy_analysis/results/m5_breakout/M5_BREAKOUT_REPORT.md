# USDCAD 5-Minute Breakout Test + EURUSD Verification — Report

> **Prepared 12 Sep 2026 · data window 1 Sep 2025 → 11 Sep 2026** · engine: the repo's own `execute_setup` (2-pip SL buffer, close-breakout entry, SL-first intrabar, EOD flat) · R = pnl_pips / risk_pips (partial-R convention used across the repo) · all times GMT (broker export time, same convention as every earlier repo result)

## What was asked

1. **USDCAD:** keep the Monday 15:45 GMT 15-min MOTHER candle but enter on the **5-minute close-breakout** (was 15-min), with **RR 1:2** (was 1:3) — does it improve the setup? — backtest **and** forward test.
2. **EURUSD:** re-verify the Thursday 11:30 GMT setup (15-min mother + 5-min breakout, RR 1:2) on the newly uploaded full-year M5 export, because the earlier EUR backtest was suspected to be corrupted by a data bug.

## TL;DR verdict

| Question | Answer |
|---|---|
| Was the earlier EURUSD forward (front) test correct? | **YES — verified correct.** The old EUR M5 file is genuine 5-min data and bar-for-bar identical to the new upload (44,913/44,913 bars). Its results reproduce exactly (31 trades, netR +12.7, PF 1.90, +13.2% @1%). The known corruption was in the OLD **in-sample** file (it was actually M15) which is no longer in the repo — the new genuine M5 file now gives a correct backtest too. |
| Does the 5-min breakout improve USDCAD? | **NO.** Forward: netR falls from **−1.68R to −3.62R** and WR from 46.9% to 43.8%. Backtest: netR falls from +2.27R to +0.44R. |
| Does RR 1:3 → 1:2 help USDCAD? | **NO.** Forward: netR falls from −1.68R to −3.20R (RR-only change, same entries). |
| Combined change (5-min + RR2, the asked plan) | **The worst of the four USDCAD variants on the forward window** (−3.62R, PF 0.77). |
| Keep EURUSD as is? | **YES** — 5-min RR2 is confirmed: +13.0R backtest / +12.7R forward (very stable), better than its own 15-min RR2 variant on the forward window (+10.6R). |

---

## 1. Data audit (done BEFORE any backtest)

| File | Rows | Coverage | Modal bar gap | Verdict |
|---|---|---|---|---|
| USDCAD M5 **new** (`..._202509011740_202609112055.htm`) | 76,859 | 2025-09-01 17:40 → 2026-09-11 20:55 | 5 min (99.9%) | ✅ genuine M5 |
| EURUSD M5 **new** (`..._202509011740_202609112055.htm`) | 76,851 | 2025-09-01 17:40 → 2026-09-11 20:55 | 5 min (99.9%) | ✅ genuine M5 |
| EURUSD M5 **old** (`..._202602012205_202609080000.htm`) | 44,913 | 2026-02-01 22:05 → 2026-09-08 00:00 | 5 min (99.9%) | ✅ genuine M5 |
| EURUSD M15 | 14,988 | 2026-02-01 22:00 → 2026-09-08 00:00 | 15 min (99.8%) | ✅ genuine M15 |
| USDCAD M15 | 15,063 | 2026-02-01 22:00 → 2026-09-08 18:45 | 15 min (99.8%) | ✅ genuine M15 |

Cross-checks:

- **Old vs new EUR M5 export:** all **44,913 overlapping bars IDENTICAL** (OHLC exact) → the earlier forward test used exactly the same prices as the new upload.
- **M5 → M15 rebuild vs real M15:** EURUSD **14,988/14,988 exact**; USDCAD **15,062/15,063 exact** (the single difference is the final PARTIAL M15 candle of the export, 8 Sep 18:45, a Tuesday — it cannot affect the Monday setups). Same timezone, same instrument → the M5 files line up perfectly with the M15 mother-candle files.
- No duplicate timestamps; weekend bars are only Sunday 21:00–23:59 market-open bars, which the plan's day filter skips anyway.

## 2. Was the earlier EURUSD test buggy? — investigation result

The repo's own `EU_GU_RISK_REPORT.md` notes that the old **in-sample** (pre-Feb-2026) data files were mislabeled — the in-sample "EURUSD M5" file was actually M15 data. That file is gone from the repo; the forward-window M5 file that produced the published EUR numbers is NOT it. Proof, in order:

1. **GATE 1** — old forward EUR M5 export ≡ new EUR M5 export, bar for bar: **44,913/44,913 identical**.
2. **GATE 2** — the new M5 file aggregates to exactly the real M15 export: 14,988/14,988 candles identical → the new M5 file is genuine, correctly-labelled, same-timezone EURUSD data.
3. **Reproduction** — the Thursday 11:30 GMT, 5-min-breakout, RR2 setup over the forward window gives **31 trades, 16 W / 15 L, WR 51.6%, netR +12.70, PF 1.90, +13.17% at 1% risk** — matching the published repo numbers (31 trades, +13.2%, PF 1.90) to the decimal.

**Verdict: the EURUSD FORWARD (front) test was CORRECT. No bug affects it.** The bug the memory refers to was in the removed in-sample file. With the new genuine full-year M5 data we can now also run the EUR **backtest** properly for the first time — see §5.

## 3. Validation gates used for THIS report (anti-bug design)

| Gate | Check | Result |
|---|---|---|
| 1 | old EUR M5 ≡ new EUR M5 (overlap, OHLC exact) | PASS 44,913/44,913 |
| 2a | USDCAD M5-rebuilt M15 ≡ real M15 | PASS 15,062/15,063 |
| 2b | EURUSD M5-rebuilt M15 ≡ real M15 | PASS 14,988/14,988 |
| 3a | USDCAD trades(real M15) ≡ trades(rebuilt M15), forward window | PASS 32 vs 32 identical |
| 3b | USDCAD 5min/RR2 trades(real) ≡ trades(rebuilt) | PASS 32 vs 32 identical |
| 3c | EURUSD 5min/RR2 trades(real) ≡ trades(rebuilt) | PASS 31 vs 31 identical |
| 4 | Baseline reproduction: USDCAD 15min/RR3 forward = earlier repo numbers | PASS 32 trades, netR −1.7 |
| 5 | Baseline reproduction: EUR 5min/RR2 forward = earlier repo numbers | PASS 31 trades, netR +12.7 |

Two harness bugs were caught and fixed by these gates during the build (documented for transparency): (a) a missing weekday filter that would have traded the Monday setup on every weekday (157 instead of 32 trades) — caught by the baseline check; (b) a naive trade-equality check that ignored the real M15 export's truncated boundary days — fixed by comparing only days where the real export actually contains the mother candle. The backtest window has no real M15 export, so the mother candle there is rebuilt from the M5 bars — this is licensed by Gates 2 and 3 (identical trades where both exist).

## 4. USDCAD — results (Monday 15:45 GMT mother candle)

Four variants isolate each change: entry timeframe (15-min vs 5-min breakout) and RR (3 vs 2). Mother candle, 2-pip SL buffer, one trade per day, same filters (ORB range 2–100 pips) in all four.

### 4a. FORWARD TEST — 1 Feb → 8 Sep 2026 (real M15 exports; identical protocol to every earlier published number)

| Variant | Trades | W/L | WR | netR | PF | Avg win R | Avg loss R | Ret @1% | MaxDD @1% | Max consec L | EOD exits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A — 15-min breakout, RR 1:3 (current plan) | 32 | 15/17 | 46.9% | -1.68 | 0.88 | +0.84 | -0.84 | -1.84% | 4.00% | 3 | 19 |
| B — 15-min breakout, RR 1:2 | 32 | 15/17 | 46.9% | -3.20 | 0.78 | +0.74 | -0.84 | -3.28% | 4.00% | 3 | 18 |
| C — 5-min breakout, RR 1:3 | 32 | 14/18 | 43.8% | -3.52 | 0.77 | +0.85 | -0.86 | -3.63% | 5.17% | 3 | 18 |
| D — **5-min breakout, RR 1:2 (ASKED)** | 32 | 14/18 | 43.8% | -3.62 | 0.77 | +0.84 | -0.86 | -3.71% | 4.54% | 3 | 16 |

### 4b. BACKTEST — 1 Sep 2025 → 31 Jan 2026 (mother candle rebuilt from the new M5 data, gate-validated)

| Variant | Trades | W/L | WR | netR | PF | Avg win R | Avg loss R | Ret @1% | MaxDD @1% | Max consec L | EOD exits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A — 15-min breakout, RR 1:3 (current plan) | 21 | 12/9 | 57.1% | +2.27 | 1.31 | +0.80 | -0.81 | +2.20% | 1.99% | 2 | 14 |
| B — 15-min breakout, RR 1:2 | 21 | 12/9 | 57.1% | +2.57 | 1.35 | +0.82 | -0.81 | +2.50% | 1.99% | 2 | 12 |
| C — 5-min breakout, RR 1:3 | 21 | 11/10 | 52.4% | +1.54 | 1.17 | +0.96 | -0.91 | +1.41% | 2.26% | 2 | 11 |
| D — **5-min breakout, RR 1:2 (ASKED)** | 21 | 11/10 | 52.4% | +0.44 | 1.05 | +0.86 | -0.91 | +0.33% | 2.26% | 2 | 10 |

21 of 21 Mondays traded (the 1 Sep 2025 export starts 17:40, after the 15:45 candle). The forward window has 32 of 32 Mondays.

### 4c. Why the new variant loses — the matched-date anatomy

**Forward window (all 32 dates traded by both) — every W/L flip:**

- **2026-05-11** (DIRECTION FLIP): old LONG +0.92R (EOD Close) → new SHORT -1.00R (SL Hit)

**Backtest window (all 21 dates traded by both) — every W/L flip:**

- **2025-12-08**: old LONG +0.15R (EOD Close) → new LONG -0.05R (EOD Close)
- **2026-01-12**: old LONG -0.03R (EOD Close) → new LONG +0.14R (EOD Close)
- **2026-01-26** (DIRECTION FLIP): old LONG +0.58R (EOD Close) → new SHORT -1.00R (SL Hit)

Mechanics behind the diffs (from the matched-date CSVs):

- The 5-min trigger closes **17 min earlier on average forward / 20 min earlier in the backtest** (median 5–10 min; identical-instant on 10/32 and 7/21 trades). Entering earlier sometimes lands on the WRONG side before the real 15-min signal forms — the direction flipped on 3 of 32 forward Mondays (11 May cost a won trade: old LONG +0.92R EOD → new SHORT −1R; 1 Jun and 7 Sep flipped loss-to-loss) and on 2 of 21 backtest Mondays (26 Jan: old LONG +0.58R → new SHORT −1R; 20 Oct: loss-to-loss).
- With RR 1:2 the TP sits closer, so big EOD winners get capped at +2.0R — 13 Apr (+2.51R→+2.0R) and 27 Apr (+3.0R→+2.0R) forward, 17 Nov (+1.6R→+2.0R, helped) and 1 Dec (+2.1R→+2.0R) backtest. Net effect forward: −1.51R from the two caps, only partly won back.
- USDCAD's core problem is unchanged by the variant: realized avg win (+0.84R) ≈ realized avg loss (−0.86R) with sub-50% WR, and 19 of 32 forward trades exit at EOD without reaching TP or SL.

### 4d. Monthly netR (USDCAD)

- A current (15min/RR3), forward: 2026-02: -0.01R (4t), 2026-03: -3.54R (5t), 2026-04: +5.38R (4t), 2026-05: +0.51R (4t), 2026-06: -2.44R (5t), 2026-07: +1.17R (4t), 2026-08: -1.75R (5t), 2026-09: -1.00R (1t)
- D asked (5min/RR2), forward: 2026-02: -0.29R (4t), 2026-03: -3.46R (5t), 2026-04: +4.01R (4t), 2026-05: -1.41R (4t), 2026-06: -3.19R (5t), 2026-07: +2.86R (4t), 2026-08: -1.14R (5t), 2026-09: -1.00R (1t)
- A current (15min/RR3), backtest: 2025-09: +0.20R (4t), 2025-10: -0.04R (4t), 2025-11: +0.48R (4t), 2025-12: +0.49R (5t), 2026-01: +1.15R (4t)
- D asked (5min/RR2), backtest: 2025-09: +0.92R (4t), 2025-10: -0.71R (4t), 2025-11: +0.30R (4t), 2025-12: +0.19R (5t), 2026-01: -0.25R (4t)

## 5. EURUSD — verification + proper backtest (Thursday 11:30 GMT, 5-min breakout, RR 1:2)

### 5a. FORWARD TEST — 1 Feb → 8 Sep 2026 (reproduces the published numbers exactly — see §2)

| Variant | Trades | W/L | WR | netR | PF | Avg win R | Avg loss R | Ret @1% | MaxDD @1% | Max consec L | EOD exits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E — 5-min breakout, RR 1:2 (current plan) | 31 | 16/15 | 51.6% | +12.70 | 1.90 | +1.67 | -0.94 | +13.17% | 2.05% | 3 | 4 |
| F — 15-min breakout, RR 1:2 (reference) | 31 | 15/16 | 48.4% | +10.62 | 1.74 | +1.66 | -0.89 | +10.86% | 4.08% | 5 | 5 |

### 5b. BACKTEST — 1 Sep 2025 → 31 Jan 2026 (first correct EUR backtest: genuine M5 data)

| Variant | Trades | W/L | WR | netR | PF | Avg win R | Avg loss R | Ret @1% | MaxDD @1% | Max consec L | EOD exits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E — 5-min breakout, RR 1:2 (current plan) | 20 | 11/9 | 55.0% | +13.00 | 2.44 | +2.00 | -1.00 | +13.58% | 3.00% | 3 | 0 |
| F — 15-min breakout, RR 1:2 (reference) | 20 | 15/5 | 75.0% | +23.05 | 5.61 | +1.87 | -1.00 | +25.56% | 1.99% | 2 | 2 |

20 of 22 Thursdays traded (25 Dec + 1 Jan holidays absent). The current 5-min plan is strikingly stable across regimes: **+13.0R backtest / +12.7R forward**, PF 2.44 / 1.90 — while its 15-min-breakout reference was spectacular in-sample (+23.1R, WR 75%) but regressed to +10.6R forward. The 5-min entry is the robust choice for EUR — the earlier design decision is confirmed.

### 5c. Front-test continuation — 9 → 11 Sep 2026 (new-data tail)

- **Thursday 10 Sep 2026: EURUSD LONG, TP hit, +2.00R** (full winner, both entry modes).
- USDCAD: no Monday in the tail (next Monday is 14 Sep, beyond the data) — nothing to report.

## 6. Conclusions & recommendation

1. **Do NOT switch USDCAD to the 5-min breakout with RR 1:2.** It is the worst of the four combinations on the forward window (−3.62R vs −1.68R for the current plan) and no better in the backtest (+0.44R vs +2.27R).
2. **RR 1:3 → 1:2 alone also hurts** (−1.68R → −3.20R forward): USDCAD's winners are mostly EOD partials (avg +0.84R), so trimming the TP trims the payoff without adding wins.
3. **The 5-min breakout alone also hurts** (−1.68R → −3.52R forward): it front-runs the 15-min signal into opposite-direction false breakouts five times across the two windows, each a full −1R.
4. USDCAD Monday remains the weak leg of the plan (netR −1.7 forward at the CURRENT settings) — consistent with the earlier EU+GU report that dropped it from the keeper combination. If anything, consider removing USDCAD rather than re-engineering its entry.
5. **EURUSD Thursday (5-min, RR 1:2) is verified correct and robust** — keep it exactly as it is. Both its backtest (+13.0R, PF 2.44) and forward (+12.7R, PF 1.90) windows are strong and consistent.

**Sample-size honesty:** 21–32 trades per variant per window. The USDCAD ranking is consistent across both independent windows (forward AND backtest agree that variant D ≤ A), but single-week trade counts are small — treat margins < 1.5R as noise.

## 7. Files

- `equity_comparison.html` — interactive chart: forward + backtest equity curves, all variants
- `summary.json` — machine-readable stats for every variant × window
- `trades_*.csv` — full trade lists (`_bt` backtest, `_fw` forward, `_ext` 9–11 Sep tail; `reb` = M5-rebuilt mother candle)
- `USDCAD_old_vs_new_fw.csv` / `_bt.csv` — matched-date comparison
- `monthly_*.csv` — monthly netR tables
- `../data_audit.py` — rerunnable data audit (`python3 strategy_analysis/data_audit.py`)
- `../m5_breakout_test.py` — rerunnable test (`python3 strategy_analysis/m5_breakout_test.py`); it hard-fails without producing results if any validation gate breaks

