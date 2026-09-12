"""
expansion_cashflow.py — BUY-FIRST / SAVE-THE-LEFT expansion economics
=====================================================================
The user's plan (verified against his FYFX receipt + the official price page):

  * EVERY payout: the account price tag comes off the top ("BUY FIRST"),
    the LEFTOVER is living/savings money ("LEFT WILL BE SAVED").
  * Purchases climb the size ladder 5K -> 10K -> 25K -> 50K -> 100K -> 200K
    (Classic + add-ons each time). Payouts are far bigger than the price
    tags, so most of each payout is saved.
  * Fee refunds (after each account's 3rd payout) come back as CASH and feed
    the same buy-first policy — this applies to ALL purchased accounts too.
  * Every purchased account trades the same EU+GU signals and runs the full
    FYFX machinery (8% target once, $150 min payout, 90% split, 6-day
    spacing, internal FYFX scaling 1x->1.5x->2x->5x->12x->30x of its size,
    8% static max loss, breach -> rebuy 5K from cash).

Prices (official page, current promo; calibrated to the user's receipt):
  promo base:      5K $45 | 10K $81 | 25K $148 | 50K $373 | 100K $720 | 200K $1,125
  add-ons (net):   20% of the LIST price less the 10% bundle discount
                   -> $17.82 at 5K (receipt: paid $64.30 all-in incl. fees)
  fee refund:      100% of the net price paid (assumption: add-ons included;
                   if FYFX refunds only the base fee, subtract the addon part)

Outputs (real 13-month verified trades + 4y Monte Carlo):
  * purchase ledger + savings curve (the "can I live on it" answer)
  * fleet overview, out-of-pocket vs received vs saved
  * fan chart of 4-year savings under the same policy

Run: python3 strategy_analysis/expansion_cashflow.py
"""
import os
import sys
import json
from datetime import date, timedelta

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eu_gu_funded_plan import (load_gbp_upload, run_window_setup, rebuild_m15,
                               EU_MOTHER, GU_MOTHER, DATA_START, FW_END,
                               parse_mt5_htm)
from funded_plan import (load_pairs, generate_core3_signals, setup_key, r_of,
                         SPLIT, STATIC_DD)

OUT = os.path.join(HERE, "strategy_analysis", "results", "expansion_cashflow")
os.makedirs(OUT, exist_ok=True)

RISK = 0.015
MAX_FLEET = 40
SIM_END = date(2026, 9, 8)

# ---- official FYFX price tags (Classic, promo prices off the live page) ---- #
SIZES = [5000, 10000, 25000, 50000, 100000, 200000]
LIST_PRICE = {5000: 99, 10000: 179, 25000: 329, 50000: 829,
              100000: 1599, 200000: 2499}
PROMO_PRICE = {5000: 45, 10000: 81, 25000: 148, 50000: 373,
               100000: 720, 200000: 1125}
ADDON_FRAC_OF_LIST = 0.20     # calibrates to the $20 add-on at 5K
BUNDLE_DISC = 0.10            # "Add-Ons Bundle Save 10%"


def net_price(size):
    return PROMO_PRICE[size] + ADDON_FRAC_OF_LIST * LIST_PRICE[size] * \
        (1 - BUNDLE_DISC)


def scale_ladder(size):
    """FYFX internal scaling: 1x -> 1.5x -> 2x -> 5x -> 12x -> 30x."""
    return [int(size * m) for m in (1, 1.5, 2, 5, 12, 30)]


# --------------------------------------------------------------------------- #
# Build the verified 13-month trade sequence (EU full-year + GU full-year)
# --------------------------------------------------------------------------- #
def build_seq():
    pairs = load_pairs()
    df = generate_core3_signals(pairs)
    df = df.copy()
    df["setup"] = df.apply(setup_key, axis=1)
    df["R"] = df.apply(r_of, axis=1)
    sub = df[df.setup.isin(["Wednesday | GBPUSD | 08:30 PM",
                            "Thursday | EURUSD | 05:00 PM"])]
    sub = sub.sort_values("entry_time")
    eu5 = parse_mt5_htm(os.path.join(
        HERE, "EURUSD_M5_202509011740_202609112055.htm"))
    eu_full = run_window_setup(EU_MOTHER, eu5, rebuild_m15(eu5),
                               DATA_START, FW_END)
    eu_full["R"] = eu_full.apply(r_of, axis=1)
    gbp5, kind, path = load_gbp_upload()
    gu_fy = run_window_setup(GU_MOTHER, None, gbp5, DATA_START, FW_END)
    gu_fy["R"] = gu_fy.apply(r_of, axis=1)
    full = pd.concat([eu_full, gu_fy]).sort_values("entry_time").reset_index(
        drop=True)
    days = pd.to_datetime(full["entry_time"]).dt.date
    return list(zip(days, full["R"].astype(float).round(4)))


# --------------------------------------------------------------------------- #
# Event-day simulation: fleet + trader cash + buy-first policy
# --------------------------------------------------------------------------- #
TARGET_FIRST = 0.08
MIN_PAYOUT_YOU = 150.0
REFUND_AT = 3
SCALE_EVERY = 3
MIN_DAYS_BTWN = 6
DAILY_DD = 0.04


class Account:
    _next_id = 1

    def __init__(self, start_size, start_day):
        self.id = Account._next_id
        Account._next_id += 1
        self.ladder = scale_ladder(start_size)
        self.level = self.ladder[0]
        self.balance = float(self.level)
        self.profit = 0.0
        self.days = 0
        self.payout_no = 0
        self.target_met = False
        self.breached = False
        self.start_day = start_day
        self.price_paid = None          # set at purchase time
        self.payouts = []               # (date, you)
        self.curves = []                # (date, balance, level)

    def take(self, d, R, risk):
        pre = self.balance
        pnl = risk * self.level * R
        self.balance += pnl
        self.profit += pnl
        self.days += 1
        self.curves.append((d, self.balance, self.level))
        ev = {"date": d, "payout": None, "refund": False, "breach": False}
        if pnl < -DAILY_DD * pre:
            ev["daily_soft"] = True
        if self.balance < self.level * (1 - STATIC_DD):
            self.breached = True
            ev["breach"] = True
            return ev
        if not self.target_met:
            elig = self.profit >= TARGET_FIRST * self.level
        else:
            elig = SPLIT * self.profit >= MIN_PAYOUT_YOU
        if elig and self.days >= MIN_DAYS_BTWN:
            self.payout_no += 1
            you = round(SPLIT * self.profit, 2)
            self.payouts.append((d, you))
            self.curves[-1] = (d, float(self.level), self.level)  # EOD reset
            scale = self.payout_no % SCALE_EVERY == 0
            if scale:
                self.level = self.ladder[min(
                    self.payout_no // SCALE_EVERY, len(self.ladder) - 1)]
                self.balance = float(self.level)
            else:
                self.balance = float(self.level)
            self.target_met = True
            self.profit = 0.0
            self.days = 0
            ev["payout"] = you
            ev["refund"] = (self.payout_no == REFUND_AT)
        return ev


def simulate(seq, start_cash=0.0, mc_R=None, seed_rng=None, horizon_end=None,
             policy="risk_aware"):
    """Run the fleet + cash policy over `seq` (or MC draws of the same dates)."""
    dates = [d for d, _ in seq]
    if mc_R is not None:
        stream = list(zip(dates, mc_R))
    else:
        stream = list(seq)
    end_day = horizon_end or stream[-1][0]

    cash = float(start_cash)
    fleet = []                       # active Account objects
    pending_opens = []               # (open_day, size)
    purchases = []                   # (date, size, price)
    ladder_idx = 0                   # next rung to buy
    ledger = []                      # rows for the cash ledger
    breach_rebuys = 0
    out_of_pocket = 0.0

    def required_cash(size):
        price = net_price(size)
        if policy == "price_only":
            return price
        # risk-aware: price + ONE full max-loss buffer of the new account
        # (8% of its level) so a worst-case first week can't wipe the wallet
        return price + 0.08 * size

    def try_buy(d):
        nonlocal ladder_idx, cash, out_of_pocket
        while ladder_idx < len(SIZES) and len(fleet) + len(pending_opens) < MAX_FLEET:
            size = SIZES[ladder_idx]
            price = net_price(size)
            if cash >= required_cash(size):
                cash -= price
                if out_of_pocket == 0.0:
                    out_of_pocket = price      # first buy = own money
                purchases.append((d, size, round(price, 2)))
                nxt = min([t for t, _ in stream if t > d] or [None],
                          key=lambda t: t)
                if nxt is not None and nxt <= end_day:
                    pending_opens.append((nxt, size))
                ladder_idx += 1
                ledger.append((d, "BUY", -price, cash, f"{size//1000}K acct"))
            else:
                break

    # account #1 bought on day 0 with own money
    first_price = net_price(SIZES[0])
    cash -= first_price
    out_of_pocket = first_price
    purchases.append((stream[0][0], SIZES[0], round(first_price, 2)))
    pending_opens.append((stream[0][0], SIZES[0]))
    ladder_idx = 1
    ledger.append((stream[0][0], "BUY", -first_price, cash, "5K acct (#1)"))

    for d, R in stream:
        if d > end_day:
            break
        # opens scheduled for today
        for od, size in [p for p in pending_opens if p[0] == d]:
            a = Account(size, d)
            a.price_paid = net_price(size)
            fleet.append(a)
        pending_opens = [p for p in pending_opens if p[0] != d]

        # every open account takes every signal
        for a in fleet:
            if a.breached:
                continue
            ev = a.take(d, R, RISK)
            if ev["payout"] is not None:
                cash += ev["payout"]
                ledger.append((d, "PAYOUT", ev["payout"], cash,
                               f"acct#{a.id} P#{a.payout_no}"))
                if ev["refund"]:
                    cash += a.price_paid
                    ledger.append((d, "REFUND", a.price_paid, cash,
                                   f"acct#{a.id} fee back"))
                try_buy(d)
            if ev["breach"]:
                ledger.append((d, "BREACH", 0.0, cash, f"acct#{a.id} dead"))
                breach_rebuys += 1
                # rebuy policy: fresh 5K from cash when affordable
                price = net_price(5000)
                if cash >= price + (0.08 * 5000 if policy == "risk_aware" else 0):
                    cash -= price
                    purchases.append((d, 5000, round(price, 2)))
                    nxt = min([t for t, _ in stream if t > d] or [None])
                    if nxt is not None and nxt <= end_day:
                        pending_opens.append((nxt, 5000))
                    ledger.append((d, "BUY", -price, cash, "5K rebuy"))

    total_received = sum(r["amt"] for r in _ledger_rows(ledger)
                         if r["kind"] in ("PAYOUT", "REFUND"))
    total_spent = -sum(r["amt"] for r in _ledger_rows(ledger)
                       if r["kind"] == "BUY")
    funded = sum(a.level for a in fleet)
    bal = sum(a.balance for a in fleet)
    return {"cash": round(cash, 2), "purchases": purchases,
            "ledger": ledger, "fleet": fleet, "breach_rebuys": breach_rebuys,
            "total_received": round(total_received, 2),
            "total_spent": round(total_spent, 2),
            "out_of_pocket": round(out_of_pocket, 2),
            "funded": funded, "balance": round(bal, 2)}


def _ledger_rows(ledger):
    return [{"date": d, "kind": k, "amt": a, "cash": c, "note": n}
            for d, k, a, c, n in ledger]


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("EXPANSION CASH-FLOW — BUY FIRST, SAVE THE LEFT (official prices)")
    print("=" * 78)
    print("\nNet account prices (promo + add-ons net of 10% bundle):")
    for s in SIZES:
        print(f"  {s//1000:>3}K: base ${PROMO_PRICE[s]:>5,} + addons "
              f"${ADDON_FRAC_OF_LIST*LIST_PRICE[s]*0.9:>6,.2f} = "
              f"${net_price(s):>7,.2f}   (list ${LIST_PRICE[s]+ADDON_FRAC_OF_LIST*LIST_PRICE[s]:,.0f})")

    seq = build_seq()
    print(f"\nVerified trade stream: {len(seq)} trades, {seq[0][0]} -> "
          f"{seq[-1][0]}")
    res_risk = simulate(seq, policy="risk_aware")
    res_aggr = simulate(seq, policy="price_only")
    res = res_risk            # plan of record = risk-aware

    for name, r in (("RISK-AWARE (price + 1 max-loss buffer)", res_risk),
                    ("AGGRESSIVE (price tag only)", res_aggr)):
        print(f"\n--- 13-MONTH RESULT (real trades): {name} ---")
        print(f"Out-of-pocket (account #1): ${r['out_of_pocket']:,.2f}")
        print(f"Total received (payouts + refunds): ${r['total_received']:,.2f}")
        print(f"Total spent on accounts: ${r['total_spent']:,.2f}")
        print(f"Purchases: {len(r['purchases'])} accounts")
        for d, size, price in r["purchases"]:
            print(f"   {d}  buy {size//1000:>3}K  ${price:>8,.2f}")
        print(f"Fleet now: {len(r['fleet'])} accounts, combined funded "
              f"${r['funded']:,}, cycle balance "
              f"${r['balance'] - r['funded']:,.2f}")
        print(f"** CASH SAVED (living money): ${r['cash']:,.2f} **")
        print(f"Breaches: {r['breach_rebuys']}")

    # ledger CSV
    pd.DataFrame(_ledger_rows(res_risk["ledger"])).to_csv(
        os.path.join(OUT, "cash_ledger.csv"), index=False)

    # ---------------- 4-year Monte Carlo of SAVINGS ---------------- #
    print("\n--- 4-YEAR MONTE CARLO of the same policy (1,000 paths) ---")
    dates = [d for d, _ in seq]
    # extend the calendar 4 years at the same pace (Wed+Thu)
    all_dates = []
    d = dates[-1] + timedelta(days=1)
    while len(all_dates) < 3 * 418:      # 3 more years after the first
        if d.weekday() in (2, 3) and not (d.month == 12 and d.day == 25) \
                and not (d.month == 1 and d.day == 1):
            all_dates.append(d)
        d += timedelta(days=1)
    R = np.array([r for _, r in seq], dtype=float)
    mu = R.mean()

    def block_draw(n, rng, block=8):
        out = np.empty(n)
        i = 0
        while i < n:
            s = rng.integers(0, len(R))
            for k in range(block):
                if i >= n:
                    break
                out[i] = R[(s + k) % len(R)]
                i += 1
        return out

    rng = np.random.default_rng(20260914)
    PATHS = 1000
    combos = [("full_edge_risk_aware", "full_edge", "risk_aware"),
              ("full_edge_aggressive", "full_edge", "price_only"),
              ("no_edge_risk_aware", "no_edge", "risk_aware")]
    finals = {k: [] for k, _, _ in combos}
    for key, scen, pol in combos:
        for p in range(PATHS):
            draw = block_draw(len(dates) + len(all_dates), rng)
            if scen == "no_edge":
                draw = draw - mu        # zero expectancy, keep variance
            stream = list(zip(dates + all_dates, draw))
            r = simulate(stream, policy=pol)
            finals[key].append((r["cash"], len(r["purchases"]),
                                r["funded"]))
    rows = []
    for key in finals:
        arr = finals[key]
        cash = np.array([x[0] for x in arr])
        accs = np.array([x[1] for x in arr])
        fund = np.array([x[2] for x in arr])
        rows.append({"scenario": scen, "median_saved": float(np.median(cash)),
                     "p5_saved": float(np.percentile(cash, 5)),
                     "p25_saved": float(np.percentile(cash, 25)),
                     "p75_saved": float(np.percentile(cash, 75)),
                     "p95_saved": float(np.percentile(cash, 95)),
                     "p_saved_under_10k": float(np.mean(cash < 10000)),
                     "median_accounts": float(np.median(accs)),
                     "median_funded": float(np.median(fund))})
        r0 = rows[-1]
        print(f"  {key:<22}: 4y savings median ${r0['median_saved']:>10,.0f} "
              f"(p5 ${r0['p5_saved']:>9,.0f} / p95 ${r0['p95_saved']:>10,.0f}) "
              f"P(saved<$10K)={r0['p_saved_under_10k']*100:.1f}%  "
              f"accounts={r0['median_accounts']:.0f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mc_savings.csv"), index=False)

    # ---------------- chart ---------------- #
    led = pd.DataFrame(_ledger_rows(res["ledger"]))
    led["date"] = pd.to_datetime(led["date"])
    buys = led[led.kind == "BUY"]
    ins = led[led.kind.isin(["PAYOUT", "REFUND"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ins.date, y=ins.cash.cumsum() * 0 + ins.cash,
                             mode="lines", name="Cash on hand after event",
                             line=dict(color="#1a6ee0", width=2)))
    fig.add_trace(go.Scatter(x=buys.date, y=buys.cash, mode="markers",
                             marker=dict(symbol="triangle-down", size=11,
                                         color="#d62728"),
                             name="Account purchase",
                             hovertemplate="%{x|%d %b %Y}<br>Cash after buy: "
                                           "$%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.update_layout(
        title=("BUY FIRST, SAVE THE LEFT — cash on hand at every payout / "
               "refund / purchase<br><sup>Real 13-month verified trades · "
               "official FYFX promo prices + add-ons (10% bundle) · each "
               "purchase climbs 5K→10K→25K→50K→100K→200K · refunds return "
               "the price paid</sup>"),
        xaxis_title="Date", yaxis_title="Trader cash (USD)",
        hovermode="x unified", template="plotly_white", height=650)
    html = os.path.join(OUT, "cashflow_chart.html")
    fig.write_html(html, include_plotlyjs=True)

    # ---------------- summary json ---------------- #
    json.dump({"prices": {f"{s//1000}K": round(net_price(s), 2)
                          for s in SIZES},
               "deterministic_13m": {
                   "out_of_pocket": res["out_of_pocket"],
                   "total_received": res["total_received"],
                   "total_spent_on_accounts": res["total_spent"],
                   "cash_saved": res["cash"],
                   "purchases": [{"date": str(d), "size": s, "price": p}
                                 for d, s, p in res["purchases"]],
                   "fleet": len(res["fleet"]), "funded": res["funded"],
                   "breaches": res["breach_rebuys"]},
               "mc_4y_savings": rows},
              open(os.path.join(OUT, "expansion_summary.json"), "w"),
              indent=2, default=str)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
