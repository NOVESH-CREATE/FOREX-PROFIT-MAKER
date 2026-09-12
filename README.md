# FOREX-PROFIT-MAKER — FINAL Funded Plan: EURUSD + GBPUSD ONLY @ 1.5% risk

> **Plan of record — synced 13 Sep 2026.** Two assets, two trades a week, 1.5%
> of the account's **current** level per trade, on FundYourFX Classic accounts
> with add-ons (90% split from day one, 8% max loss). Everything USDCAD-related
> and the old core-3 / 1%-risk material elsewhere in the repo is **historical
> evidence**, not the plan.

## The plan — 2 trades/week, 2 assets

| Day | Pair | ORB candle (GMT) | Entry | RR |
|---|---|---|---|---|
| Wednesday | GBPUSD | 15:00 | 15-min close-breakout | 1:3 |
| Thursday | EURUSD | 11:30 | 5-min close-breakout | 1:2 |

- **2 trades/week** = 104 trades/yr ≈ 2.0/week — no other setups, no extra
  entries, 2-pip SL buffer.
- **Risk 1.5% of the account's CURRENT level per trade** — never increased by
  choice; a scaled account's dollar risk grows only because its funded level
  grew.
- **USDCAD is dropped**: 32 forward Mondays = netR **−1.68R** (PF 0.88), and
  every tested variant (5-min entry, RR 1:2) was worse — see
  `strategy_analysis/results/m5_breakout/`. The `USDCAD_*.htm` exports in the
  repo root are kept **as historical evidence only**.

## FundYourFX Classic rules (corrected from the official FYFX pages)

- **NO consistency rule** (no 25% best-day cap)
- **8% profit target, once** — unlocks payouts; not re-earned between payouts
- **Minimum $150 payout** — what you receive (≥ ~$167 gross at the 90% split)
- **Fee refund after the account's 3rd payout** — it buys a NEW $5K account
- **Per-account scaling every 3 payouts**: $5K → $7.5K → $10K → $25K → $60K → $150K
- **90% profit split** from day one (add-on) · **8% static max loss** (hard
  breach) · **4% daily drawdown** (soft, monitored) · min **6 trading days**
  between payouts

## Buy first, save the left (expansion policy — official net price tags)

Every payout first covers the next account's price tag; everything above that
is living money. Purchases climb 5K → 10K → 25K → 50K → 100K → 200K; fee
refunds come back as cash and feed the same policy. Official net prices (promo
+ add-ons at the 10% bundle discount):

| Size | 5K | 10K | 25K | 50K | 100K | 200K |
|---|---|---|---|---|---|---|
| **You pay** | $62.82 | $113.22 | $207.22 | $522.22 | $1,008.42 | $1,574.82 |

Cash-flow model on 13 months of the real verified trades: the **risk-aware**
policy (next size only when cash covers the price AND one full max-loss of that
account) completes the ladder by 6 Aug 2026 with a **$535,000 fleet** and
**$21,673.03 kept as living money** — see
`strategy_analysis/results/expansion_cashflow/EXPANSION_REPORT.md`.

## Verified results — three windows, all @ 1.5% (final engine)

| Window | Trades | Payouts | Accounts | Paid out | TOTAL PROFIT |
|---|---|---|---|---|---|
| BACKTEST Sep '25 – Jan '26 (EU + GU) | 42 (+20.66R, PF 2.59) | 6 | 2 | $1,806.59 | $2,327.41 |
| FORWARD Feb – Sep '26 — **the as-of-today reality** | 62 (+16.63R) | 5 | 2 | $1,188.11 | $1,727.79 |
| FULL YEAR Sep '25 – Sep '26 (max-history view) | 104 | 24 | **4** | **$9,411.08** | **$9,561.08** |

Full-year cascade: **4 accounts, $82,500 combined funded, $9,411.08 paid out,
$9,561.08 total profit.** Two-window stability: backtest +20.66R vs forward
+16.63R — no in-sample/out-sample flip. Full ledgers:
`strategy_analysis/results/eu_gu_1pct5/EU_GU_FUNDED_REPORT.md`.

### Verification gates (re-run on every engine run — all PASS)

| Gate | Check | Result |
|---|---|---|
| A | forward EU+GU sequence reproduces `EU_GU_RISK_REPORT.md` exactly: **+17.50% @1% / +26.90% @1.5%** | PASS |
| B | forward trade counts: **31 EUR + 31 GBP** | PASS |
| C | full-year EU sequence ≡ standard forward EU sequence on the overlap | PASS |
| D | uploaded GBPUSD full-year CSV vs the real M15 export: **14,987/14,987 timestamps** match | PASS |
| E | forward trades identical real-vs-upload: **31 v 31** | PASS |

## Run

```bash
python3 funded_plan.py        # needs: pandas, numpy, plotly
```

`funded_plan.py` is **superseded but kept**: running it now delegates to the
FINAL verified engine `strategy_analysis/eu_gu_funded_plan.py` (EU+GU only
@ 1.5%), which re-verifies gates A–E on every run and refuses to output on
drift. Its MT5 parsers (`parse_mt5_htm`, `parse_mt5_csv_like`, `load_pairs`, …)
still live in `funded_plan.py` because every analysis module imports them —
kept byte-identical so every verified result stays reproducible. The legacy
core-3 simulation is retained but must be called explicitly:
`python3 -c "import funded_plan; funded_plan.main()"`.

**Final dashboard:** `strategy_analysis/results/eu_gu_1pct5/index.html`
(interactive 3-panel chart: full-year cascade equity vs the 8% max-loss floor ·
cumulative payouts · active accounts).

## Repo layout

- `funded_plan.py` — LEGACY core-3 script; `__main__` delegates to the final engine (its MT5 parsers are kept here — every module imports them)
- `strategy_analysis/eu_gu_funded_plan.py` — **FINAL PLAN engine**: EU+GU only @ 1.5%, corrected FYFX rules, self-verifying (refuses to output on gate failure)
- `strategy_analysis/results/eu_gu_1pct5/` — **final deliverable** (`EU_GU_FUNDED_REPORT.md`, `index.html` dashboard, payout ledgers, equity CSVs, trades CSVs, `eu_gu_summary.json`)
- `strategy_analysis/backtest_engine.py` — ORB trade-execution engine (imported by everything)
- `strategy_analysis/data_audit.py` — verifies every .htm export (genuine M5/M15 spacing, old-vs-new identity, M5→M15 rebuild vs real M15)
- `strategy_analysis/expansion_cashflow.py` + `results/expansion_cashflow/` — buy-first/save-the-left cash-flow model + 4-year Monte Carlo of savings
- `strategy_analysis/project_4yr.py` + `results/projection_4yr/` — 4-year Monte Carlo projection engine
- `strategy_analysis/results/m5_breakout/` — the USDCAD post-mortem (why it was dropped)
- `strategy_analysis/results/multiacct/` — **LEGACY** core-3 @ 1% dashboard/report — bannered SUPERSEDED, kept as historical evidence
- `strategy_analysis/results/final_fixed/` — `EU_GU_RISK_REPORT.md` (the verified numbers gate A pins to) · `results/classic_addon/` — Classic add-on payout-table reference
- `EURUSD_*.htm`, `GBPUSD_*.htm/.csv` — the live-plan data (MT5 exports + the user-uploaded GBPUSD full-year M15 CSV); `USDCAD_*.htm` — historical evidence only
