# BUY FIRST, SAVE THE LEFT — expansion cash-flow report

> Confirms the user's plan: every payout first covers the next account's price tag, the rest is LIVING MONEY; purchases climb 5K → 10K → 25K → 50K → 100K → 200K; fee refunds (each account's 3rd payout) come back as cash and feed the same policy; every purchased account trades the same EU+GU signals with full FYFX rules (8% target once, $150 payouts, 90% split, internal scaling, 8% max loss).

## Net price tags (official FYFX promo + add-ons at 10% bundle discount)

| Size | Promo base | Add-ons (net) | **You pay** |
|---|---|---|---|
| 5K | $45 | $17.82 | **$62.82** |
| 10K | $81 | $32.22 | **$113.22** |
| 25K | $148 | $59.22 | **$207.22** |
| 50K | $373 | $149.22 | **$522.22** |
| 100K | $720 | $287.82 | **$1,007.82** |
| 200K | $1,125 | $449.82 | **$1,574.82** |

Calibration: the user's real receipt — 5K Classic + both add-ons = **$64.30 out of pocket** (model: $62.82 + ~fees). The first payout (~$371) covers it 6×.

## 13 months of REAL verified trades — two purchase policies

| Policy | Ladder complete by | Fleet end | Cash SAVED |
|---|---|---|---|
| **RISK-AWARE (price + 1 max-loss buffer)** — recommended | 6 Aug 2026 | $535,000 funded | **$21,673.03** |
| Aggressive (price tag only) | 27 Nov 2025 (!) | $2,055,000 funded | $204,653.04 |

- Both start with **$62.82 out of pocket** (account #1) and finish with 6 accounts, zero breaches.
- Risk-aware ledger: $25,161.15 received, $3,488.12 spent on accounts, **$21,673.03 kept as living money**.
- The aggressive variant buys a 200K account in month 3 while holding ~$600 cash — one bad streak on a $200K account (8% = $16,000) would wipe everything out. That is why the risk-aware buffer exists: buy the next size only when cash covers the price AND one full max-loss of that account.

## 4-year Monte Carlo of SAVINGS (1,000 paths, same policy)

| Scenario | Median saved | p5 | p95 | P(saved < $10K) | Accounts |
|---|---|---|---|---|---|
| full_edge_risk_aware | $32,597,547 | $4,535,372 | $71,831,401 | 1.7% | 12 |
| full_edge_aggressive | $28,047,330 | $4,395,935 | $75,082,240 | 1.5% | 15 |
| no_edge_risk_aware | $304 | $-63 | $99,975 | 72.2% | 1 |

## Read this before you believe any big number

1. The multi-million 4-year medians are the machine's arithmetic on three assumptions stacked for 4 straight years: the edge persists at +0.36R/trade, FYFX really scales every account toward 30× its size and keeps paying, and promo prices/rules stay. Real-world, treat them as the upper envelope, not a forecast.
2. The no-edge row is the honest anchor: **median $304 saved, 72% chance under $10K** — the buy-account machinery cannot print money without the edge. The edge is the business; the fleet is just leverage on it.
3. Assumptions you should verify on your dashboard: add-on pricing on bigger sizes (modelled as 20% of list, verified only at 5K), that the fee refund returns the full net price paid (add-ons included), and the internal scaling steps for purchased 10K–200K accounts (modelled as the same 1×→1.5×→2×→5×→12×→30× ladder as the 5K).
4. FYFX counterparty/rule risk is not simulatable and dwarfs everything at a 4-year horizon.

## Files

- `cash_ledger.csv` — every payout / refund / purchase with cash balance
- `cashflow_chart.html` — cash-on-hand chart with purchase markers
- `mc_savings.csv`, `expansion_summary.json` — full numbers
- Rerun: `python3 strategy_analysis/expansion_cashflow.py`

