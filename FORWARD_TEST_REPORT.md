# 📈 Forward-Test Report — ORB Scenario 3 (out-of-sample data found on GitHub)

*Prepared 2026‑09‑08 · Companion to `ORB_STRATEGY_ANALYSIS.md` and `STRATEGY_CHEATSHEET.md`*

---

## 1. What I could and couldn't do

You asked me to fetch free, gap-free data from GitHub / free sources and forward-test the 7 months
the strategy was **not** backtested on (after 2026‑02‑08).

**Result: I found real, same-broker-format MT5 data for 2 of your 3 pairs covering Feb → May 2026,
ran a genuine out-of-sample test on the 3 setups those pairs trade — and it surfaced an important
problem with your original backtest. USDCAD data (your most-traded pair) and May 24 → Sep 8 data
for the others are still needed, and there is a one-click way to get them (Section 6).**

---

## 2. First, the answer to your data/label question

I checked **row counts + actual candle spacing** of every file. Your original CSVs are real,
continuous, gap-free MT5 data — but 4 of 6 files have **wrong labels**:

| File | Rows | True candle spacing | Verdict |
|---|---|---|---|
| EURUSD `M5` (…2145) | 12,372 | **15 min** (96/day) | ⚠️ It's really M15 data, not M5 |
| EURUSD `M15` | 12,372 | 15 min | ✅ correct |
| GBPUSD `M5` (…2155) | 37,065 | **5 min** (288/day) | ✅ (the M5 is correct) |
| GBPUSD `M15` | 37,065 | **5 min** (288/day) | ⚠️ It's really M5 data, not M15 |
| USDCAD `M15` | 12,372 | 15 min | ✅ correct |
| USDCAD `M5` | 37,080 | 5 min | ✅ correct |

So: the EURUSD `M5` and the GBPUSD `M15` files are **identical copies of the M15 / M5 file**
(you exported the wrong timeframe twice). The data itself is fine — same broker feed the whole time,
5 trading days/week, ~24h days — only the file labels lied.

**Why this matters:** your Feb‑8 backtest marked the GBPUSD "15-min ORB" candle on a single
**5-minute** candle, and watched the EURUSD "5-min breakout" on **15-minute** bars. That is *not*
the strategy as written on paper.

---

## 3. Data I found on GitHub (free, same broker format)

Public mirror **`lapijoseph/Quant_models`** contains MetaTrader-5 tab-format exports of the exact
same `EURUSDm` / `GBPUSDm` / etc. symbols your repo uses:

| Pair | Available | Coverage found | Timeframes verified |
|---|---|---|---|
| EURUSD | M5 + M15 (full + 5,000-bar tail) | … → **2026‑05‑22** | true 5/15-min spacing ✅ |
| GBPUSD | M15 tail only (M5 tail also present) | **2026‑03‑11 → 2026‑05‑22** | true 15-min spacing ✅ |
| USDCAD | ❌ **not present in any public repo I could find** | — | — |

Provenance check vs your files (EURUSD M15, overlapping Aug'25–Jan'26): **every timestamp matches
exactly** (same broker server clock), and ~72% of candles are byte-identical; the other ~28% differ
by 1 pipette (0.00001–0.00004) — i.e. the same feed family, slightly different server instance.
Good enough for an out-of-sample test, but not bit-perfect.

---

## 4. ⚠️ Key finding: your backtest numbers partly came from the mislabeled files

To make the forward test meaningful, I first re-ran the engine **in-sample (Aug 2025 – Jan 2026)
on true timeframes** — GBPUSD M15 rebuilt by resampling your genuine 5-min file, EURUSD M5 from the
mirror's genuine 5-min file. Same engine, same rules, correct data:

| Setup | Your Feb-8 backtest log (mislabeled data) | Re-run on TRUE timeframes |
|---|---|---|
| Wed GBPUSD 08:30 PM (RR3) | 26 tr · **61.5%** · +38R | 26 tr · **65.4%** · +42R |
| Thu EURUSD 05:00 PM (RR2) | 24 tr · **66.7%** · +24R | 24 tr · **45.8%** · +9R |
| Thu GBPUSD 09:15 AM (RR2) | 11 tr · **63.6%** · +10R | 24 tr · **33.3%** · 0R |
| **Total (these 3)** | 61 tr · **63.9%** · +72R | 74 tr · **48.7%** · +51R |

Two of the three setups lose most of their advertised edge once tested the way the strategy was
designed (real 15-min mother candle / real 5-min breakout). The famous **+134.8%/$33.7k headline was
partly an artifact of the file mix-up.** The strategy still made money in-sample on true data (+51R),
driven mainly by Wednesday GBPUSD.

---

## 5. Forward test result (what actually happened Feb → May 2026)

Same engine, **true-timeframe** data, out-of-sample window **9 Feb 2026 → 22 May 2026**
(EURUSD from Feb; GBPUSD setups from 12 Mar, when the mirror's GBPUSD M15 begins).

### 37 real trades — 16 wins / 21 losses = **43.2% win rate, +15.0R net (+$3,000 at $200/trade)**

| Setup | Trades | Wins | Losses | Win rate | Net R |
|---|---|---|---|---|---|
| Thu EURUSD 05:00 PM (RR 1:2) | 16 | 7 | 9 | 43.8% | +5.0 |
| Thu GBPUSD 09:15 AM (RR 1:2) | 11 | 5 | 6 | 45.5% | +4.0 |
| Wed GBPUSD 08:30 PM (RR 1:3) | 10 | 4 | 6 | 40.0% | +6.0 |

(+53.8 pips total. Full trade-by-trade log: `strategy_analysis/results/forward_partial/forward_trades.csv`)

**Reading it honestly:**
- The edge **did not repeat** at anywhere near backtest strength. Forward win rate 43% vs 49% in-sample(true) vs 64% in your old log. R/trade fell from ~0.69 (true in-sample) to ~0.41.
- It stayed **above break-even** (break-even WR at these RRs ≈ 33–40%), so +15R with 0 losses-of-account risk, but that's ~0.4R/trade — a thin, fragile edge on 37 trades (error bar is roughly ±6–10R either way).
- Every setup was individually still positive in R — nothing collapsed to zero, and Wednesday GBP's RR3 kept it the best earner (only 10 trades though).

---

## 6. What's missing — and the one-click way to finish the full 7-month test

To run the **complete** 7-month forward test you need all 6 setups, i.e. add:
1. **USDCAD M5 + M15** (3 setups — Mon, Thu, Fri — are ~55% of your trade count), and
2. **May 23 → Sep 8 2026** for EURUSD/GBPUSD (and USDCAD from Feb).

Dukascopy's free feed (`datafeed.dukascopy.com`, tick-grade 1-min candles, no gaps, no API key) has
all of it. This sandbox's network can't reach it, but **any internet machine can**, and the fetcher
is already written and tested in this repo:

```bash
# on any PC / GitHub Codespaces / Colab with python3:
python3 strategy_analysis/fetch_dukascopy.py --start 2026-02-01 --end 2026-09-08 --out forward_data
git add forward_data && git commit -m "forward data" && git push   # (or upload the 6 files here)
```

It downloads all 3 pairs, resamples to true M5/M15, writes the exact MT5 tab-format files, and
auto-verifies spacing. Once the files are in `forward_data/`, I run:

```bash
python3 strategy_analysis/run_forward_test.py --data forward_data
```

and you get the full 6-setup, 7-month backtest-vs-forward comparison.

> GitHub won't let my bot account push `.github/workflows/…` files (permission), but the workflow
> `fetch_fx_data.yml` is ready locally at `.github/workflows/` — if you copy it into your repo
> (add file → paste → commit on `main`), then open the **Actions** tab → **Fetch FX forward-test data**
> → **Run workflow**, it downloads and commits the data automatically. Then tell me and I'll test.

---

## 7. Bottom line

1. Your original files were **mislabeled, not bad data** — and the mislabeling inflated the backtest (esp. Thu EURUSD & Thu GBPUSD). Verified by row counts + spacing, and by a same-period true-timeframe re-run.
2. The only free **forward** data I could find on GitHub gives **Feb–May 2026 for EURUSD & GBPUSD only**.
3. On that data the strategy **stayed profitable but degraded hard**: 43% WR, +15R on 37 trades — consistent with an overfit-to-mislabeled-data system that has *some* surviving edge.
4. For the complete answer (all 6 setups, full Feb–Sep window): fetch Dukascopy data with the one command above (or the workflow), drop the files in `forward_data/`, and the full test runs automatically.
