"""
Multi-account growth: reinvest every fee refund into a NEW Classic $5K account.

Scope: 1 Feb 2026 -> 9 Sep 2026 ONLY (no future replay).

Rules (Classic + add-ons):
  - 90% profit split from day one (flat)
  - 8% static max loss, 4% daily DD, 10% target, min 6 days
  - fee refund after 2 successful payouts -> that refund buys a NEW $5K account
    which starts trading the same core-3 signals from that day
  - scaling every 3 payouts (per account): 5k->7.5k->10k->25k->60k->150k
  - risk per trade stays 1% of the account's level (no increase)

Strategy: core-3 (Mon USDCAD RR3, Wed GBPUSD RR3, Thu EURUSD RR2), original entry.

Outputs:
  multiacct_equity.html   interactive Plotly chart (clickable / hover labels)
  multiacct_report.md     detailed analysis
  multiacct_summary.json / per-account CSVs
"""
import sys, os, json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import run_funded_sim as FS
import run_funded_native as FN

RESULTS = os.path.join(HERE, "strategy_analysis", "results")
OUT = os.path.join(RESULTS, "multiacct")

STATIC_DD = 0.08
DAILY_DD = 0.04
TARGET = 0.10
MIN_DAYS = 6
SPLIT = 0.90
LADDER = [5000, 7500, 10000, 25000, 60000, 150000]
CORE = ["Wednesday | GBPUSD | 08:30 PM", "Monday | USDCAD | 09:15 PM",
        "Thursday | EURUSD | 05:00 PM"]
START = date(2026, 2, 1)
END = date(2026, 9, 9)
RISK = 0.01


def get_core():
    pairs = FS.load_forward_pairs()
    df = FN.gen_native(pairs, "orig").copy()
    df["setup"] = df.apply(FS.setup_key, axis=1)
    sub = df[df.setup.isin(CORE)].sort_values("entry_time").reset_index(drop=True)
    sub["R"] = sub.apply(FS.r_of, axis=1)
    sub["day"] = pd.to_datetime(sub["entry_time"]).dt.date
    return sub


def simulate_account(seq, start, end, risk=RISK):
    """Simulate ONE account from `start` to `end` (trades with start<=date<=end)."""
    trades = [(d, R) for (d, R) in seq if start <= d <= end]
    events = []
    curve = []
    level_idx = 0
    level = LADDER[level_idx]
    balance = float(level)
    profit = 0.0
    best_day = 0.0
    days = 0
    soft = 0
    payout_no = 0
    breached = False
    for d, R in trades:
        risk = RISK * level
        pnl = risk * R
        balance += pnl
        profit += pnl
        days += 1
        best_day = max(best_day, pnl)
        if pnl < -DAILY_DD * (balance - pnl):
            soft += 1
        curve.append((d, balance, level))
        if balance < level * (1 - STATIC_DD):
            breached = True
            break
        if profit >= TARGET * level and best_day <= 0.25 * profit and days >= MIN_DAYS:
            payout_no += 1
            events.append({
                "payout_no": payout_no, "date": d, "level": level,
                "gross": round(profit, 2), "you": round(SPLIT * profit, 2),
                "split_pct": int(SPLIT * 100),
                "refund": payout_no == 2,
                "scale": payout_no % 3 == 0,
                "new_level": (LADDER[min(payout_no // 3, len(LADDER) - 1)]
                              if payout_no % 3 == 0 else None),
            })
            level_idx = min(payout_no // 3, len(LADDER) - 1)
            level = LADDER[level_idx]
            balance = float(level)
            profit = 0.0
            best_day = 0.0
            days = 0
            soft = 0
    return {"events": events, "curve": curve, "breached": breached,
            "n_trades": len(trades)}


def balance_on(curve, D, default=float(LADDER[0])):
    """Latest balance on/before date D."""
    val = default
    for d, b, lv in curve:
        if d <= D:
            val = b
        else:
            break
    return val


def level_on(curve, D, default=float(LADDER[0])):
    val = default
    for d, b, lv in curve:
        if d <= D:
            val = lv
        else:
            break
    return val


def main():
    os.makedirs(OUT, exist_ok=True)
    sub = get_core()
    seq = list(zip(sub["day"], sub["R"]))
    print(f"core-3 signals: {len(seq)} trades, {seq[0][0]} -> {seq[-1][0]}, netR={sum(r for _, r in seq):.1f}")

    # ---- orchestrate accounts ----
    accounts = []
    pending = [START]
    aid = 0
    while pending and aid < 40:
        sd = pending.pop(0)
        aid += 1
        res = simulate_account(seq, sd, END)
        accounts.append({"id": aid, "start": sd, **res})
        for e in res["events"]:
            if e["refund"] and e["date"] <= END:
                pending.append(e["date"])   # refund buys a new account
    print(f"\nAccounts opened by {END}: {len(accounts)}")

    # ---- aggregate time series ----
    master_dates = sorted({START} | {END} |
                          {d for a in accounts for d, b, lv in a["curve"]} |
                          {a["start"] for a in accounts})
    combined = []
    for D in master_dates:
        act = [a for a in accounts if a["start"] <= D]
        cb = sum(balance_on(a["curve"], D) for a in act)
        clv = sum(level_on(a["curve"], D) for a in act)
        combined.append({"date": D, "balance": round(cb, 2), "level": round(clv, 2),
                         "accounts": len(act)})

    # payouts + opens
    payouts = []
    for a in accounts:
        for e in a["events"]:
            payouts.append({"account": a["id"], "date": e["date"], **e})
    payouts.sort(key=lambda x: x["date"])
    opens = [{"account": a["id"], "date": a["start"]} for a in accounts]
    cum = 0.0
    for p in payouts:
        cum += p["you"]
        p["cumulative"] = round(cum, 2)

    # ---- report to stdout ----
    print("\n=== PER-ACCOUNT SUMMARY ===")
    for a in accounts:
        ev = a["events"]
        last = a["curve"][-1] if a["curve"] else (a["start"], float(LADDER[0]), float(LADDER[0]))
        paid = sum(e["you"] for e in ev)
        print(f"  Acct#{a['id']:<2} opened {a['start']}  trades={a['n_trades']:<3} "
              f"payouts={len(ev):<2} paidOut=${paid:>8,.2f}  "
              f"refundAt={next((str(e['date']) for e in ev if e['refund']),'-'):<12} "
              f"now: lvl=${last[2]:,.0f} bal=${last[1]:,.2f}  {'BREACHED' if a['breached'] else ''}")

    print("\n=== PAYOUT LEDGER (1 Feb -> 9 Sep) ===")
    for p in payouts:
        note = []
        if p["refund"]: note.append("FEE REFUND -> new account")
        if p["scale"]: note.append(f"SCALE ${p['new_level']:,}")
        print(f"  {p['date']}  Acct#{p['account']:<2} P#{p['payout_no']}  gross=${p['gross']:>7,.2f} "
              f"{p['split_pct']}% you=${p['you']:>7,.2f}  cum=${p['cumulative']:>9,.2f}  {', '.join(note)}")

    print("\n=== AS OF 9 SEP 2026 ===")
    today = combined[-1]
    total_paid = payouts[-1]["cumulative"] if payouts else 0.0
    print(f"  active accounts : {today['accounts']}")
    print(f"  combined funded : ${today['level']:,.0f}")
    print(f"  combined balance: ${today['balance']:,.2f}")
    print(f"  total paid out  : ${total_paid:,.2f}")

    # ---- save data ----
    json.dump({"accounts": [{"id": a["id"], "start": str(a["start"]), "n_trades": a["n_trades"],
                             "breached": a["breached"], "events": a["events"]} for a in accounts],
               "payouts": payouts, "opens": [{"account": o["account"], "date": str(o["date"])} for o in opens],
               "as_of_today": {"date": "2026-09-09", "accounts": today["accounts"],
                               "combined_funded": today["level"],
                               "combined_balance": today["balance"],
                               "total_paid_out": total_paid}},
              open(os.path.join(OUT, "multiacct_summary.json"), "w"), indent=2, default=str)
    pd.DataFrame(combined).to_csv(os.path.join(OUT, "combined_equity.csv"), index=False)
    for a in accounts:
        pd.DataFrame([{"date": str(d), "balance": b, "level": lv} for d, b, lv in a["curve"]]
                     ).to_csv(os.path.join(OUT, f"account_{a['id']}_equity.csv"), index=False)

    build_chart(combined, accounts, payouts, opens)
    build_report(accounts, payouts, opens, today, total_paid)
    print("\nSaved:", OUT)


# --------------------------------------------------------------------------- #
# Interactive chart
# --------------------------------------------------------------------------- #
def build_chart(combined, accounts, payouts, opens):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.68, 0.32],
                        subplot_titles=("Account equity (click/hover any point for details)",
                                        "Cumulative payouts & active accounts"),
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    cd = [r["date"] for r in combined]
    cb = [r["balance"] for r in combined]
    clv = [r["level"] for r in combined]
    ca = [r["accounts"] for r in combined]
    floor = [lv * (1 - STATIC_DD) for lv in clv]

    # combined equity
    fig.add_trace(go.Scatter(
        x=cd, y=cb, name="Combined equity", mode="lines+markers", line=dict(color="#1a6ee0", width=2),
        hovertemplate=("Date: %{x|%d %b %Y}<br>Combined balance: $%{y:,.0f}<br>"
                       "Active accounts: %{customdata[0]}<br>Combined funded: $%{customdata[1]:,.0f}"
                       "<extra></extra>"),
        customdata=list(zip(ca, clv))), row=1, col=1)

    # combined static DD floor
    fig.add_trace(go.Scatter(
        x=cd, y=floor, name="8% static max-loss floor (combined)", mode="lines",
        line=dict(color="#d62728", width=1.5, dash="dash"),
        hovertemplate="DD floor: $%{y:,.0f}<extra></extra>"), row=1, col=1)

    # per-account equity
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for a in accounts:
        if not a["curve"]:
            continue
        x = [a["start"]] + [d for d, b, lv in a["curve"]]
        y = [float(LADDER[0])] + [b for d, b, lv in a["curve"]]
        lv = [float(LADDER[0])] + [l for d, b, l in a["curve"]]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=f"Account #{a['id']}", mode="lines",
            line=dict(color=colors[(a["id"] - 1) % len(colors)], width=1.4, dash="dot"),
            hovertemplate=(f"Account #{a['id']}<br>Date: %{{x|%d %b %Y}}<br>"
                           "Balance: $%{y:,.0f}<br>Level: $%{customdata:,.0f}<extra></extra>"),
            customdata=lv), row=1, col=1)

    # cumulative payout (row2, left axis)
    px = [p["date"] for p in payouts]
    py = [p["cumulative"] for p in payouts]
    fig.add_trace(go.Scatter(
        x=px, y=py, name="Cumulative payout (you)", mode="lines+markers",
        line=dict(color="#2ca02c", width=2), fill="tozeroy",
        hovertemplate="Date: %{x|%d %b %Y}<br>Cumulative paid out: $%{y:,.0f}<extra></extra>"),
        row=2, col=1, secondary_y=False)

    # payout event markers (row2)
    fig.add_trace(go.Scatter(
        x=px, y=py, name="Payouts", mode="markers",
        marker=dict(color="#2ca02c", size=9, symbol="circle"),
        text=[f"Acct#{p['account']} P{p['payout_no']}" for p in payouts],
        hovertemplate=("Acct #%{customdata[0]} · Payout %{customdata[1]}<br>"
                       "Date: %{x|%d %b %Y}<br>Level: $%{customdata[2]:,.0f}<br>"
                       "Gross: $%{customdata[3]:,.0f} · Split %{customdata[4]}%<br>"
                       "<b>You receive: $%{customdata[5]:,.0f}</b><br>%{customdata[6]}"
                       "<extra></extra>"),
        customdata=[[p["account"], p["payout_no"], p["level"], p["gross"], p["split_pct"],
                     p["you"], ("FEE REFUND→new acct · " if p["refund"] else "") +
                     (f"SCALE→${p['new_level']:,}" if p["scale"] else "")] for p in payouts]),
        row=2, col=1, secondary_y=False)

    # account opens (row2)
    ox = [o["date"] for o in opens]
    fig.add_trace(go.Scatter(
        x=ox, y=[0] * len(ox), name="New account opened", mode="markers",
        marker=dict(color="#ff7f0e", size=11, symbol="star"),
        text=[f"Account #{o['account']}" for o in opens],
        hovertemplate=("Account #%{customdata[0]} opened<br>Date: %{x|%d %b %Y}<br>"
                       "Funded with refunded fee · level $5,000<extra></extra>"),
        customdata=[[o["account"]] for o in opens]), row=2, col=1, secondary_y=False)

    # account count (row2, right axis)
    acx = [r["date"] for r in combined]
    acy = [r["accounts"] for r in combined]
    fig.add_trace(go.Scatter(
        x=acx, y=acy, name="Active accounts", mode="lines",
        line=dict(color="#9467bd", width=1.5),
        hovertemplate="Active accounts: %{y}<extra></extra>"),
        row=2, col=1, secondary_y=True)

    fig.add_vline(x=datetime(2026, 9, 9), line_width=1.5, line_dash="dot",
                  line_color="#555", annotation_text="TODAY 9 Sep",
                  annotation_position="top left", row=1, col=1)

    fig.update_yaxes(title_text="Balance (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Payout (USD)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Accounts", row=2, col=1, secondary_y=True)
    fig.update_layout(
        title=("FundYourFX Classic $5,000 + add-ons (90% split, 8% max loss) — MULTI-ACCOUNT growth<br>"
               "<sup>Core-3 ORB · 1% risk/trade · every fee refund buys a new account · 1 Feb → 9 Sep 2026</sup>"),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=90), template="plotly_white")
    html = os.path.join(OUT, "multiacct_equity.html")
    fig.write_html(html, include_plotlyjs=True)
    print("Chart:", html)


def build_report(accounts, payouts, opens, today, total_paid):
    L = ["# Multi-Account Growth Report — Classic $5K + add-ons (1 Feb → 9 Sep 2026)",
         "",
         "> **Plan:** start ONE Classic $5K account on 1 Feb. Every time an account's fee is "
         "refunded (after its 2nd payout), do NOT withdraw it — use it to buy a NEW $5K account "
         "and trade the same core-3 signals on it from that day. Risk per trade stays **1% of the "
         "account's level** (no increase). Add-ons: **90% split from day one**, **8% max loss**.",
         "",
         "## Snapshot as of 9 Sep 2026",
         "",
         f"- **Active accounts: {today['accounts']}**",
         f"- **Combined funded capital: ${today['level']:,.0f}**",
         f"- **Combined account balance: ${today['balance']:,.2f}**",
         f"- **Total paid out to you: ${total_paid:,.2f}**",
         "",
         "## Account-by-account",
         "",
         "| Account | Opened | Signals traded | Payouts | Paid out | Fee refund | Level now | Balance now |",
         "|---|---|---|---|---|---|---|---|"]
    for a in accounts:
        ev = a["events"]
        paid = sum(e["you"] for e in ev)
        refund = next((str(e["date"]) for e in ev if e["refund"]), "—")
        last = a["curve"][-1] if a["curve"] else (a["start"], float(LADDER[0]), float(LADDER[0]))
        L.append(f"| #{a['id']} | {a['start']} | {a['n_trades']} | {len(ev)} | "
                 f"${paid:,.2f} | {refund} | ${last[2]:,.0f} | ${last[1]:,.2f} |")
    L += ["",
          "## Payout ledger (chronological)",
          "",
          "| Date | Account | Payout # | Gross | Split | You receive | Cumulative | Note |",
          "|---|---|---|---|---|---|---|---|"]
    for p in payouts:
        note = []
        if p["refund"]: note.append("**FEE REFUND → new account**")
        if p["scale"]: note.append(f"**SCALE → ${p['new_level']:,}**")
        L.append(f"| {p['date']} | #{p['account']} | {p['payout_no']} | ${p['gross']:,.2f} | "
                 f"{p['split_pct']}% | ${p['you']:,.2f} | ${p['cumulative']:,.2f} | "
                 f"{', '.join(note)} |")
    L += ["",
          "## How the accounts multiply",
          ""]
    for o in opens:
        if o["account"] == 1:
            L.append(f"- **1 Feb** — buy Account #1 with your own money ($5,000 Classic).")
        else:
            L.append(f"- **{o['date']}** — a fee refund lands → buy Account #{o['account']} "
                     f"and start trading the same signals.")
    L += ["",
          "## Interactive chart",
          "",
          "- Open `multiacct_equity.html` — it is fully clickable: hover any equity point for "
          "date/balance/accounts, hover payout dots for the full payout detail (gross, split, your $), "
          "hover stars for account openings, and click legend items to toggle traces on/off.",
          "",
          "## Assumptions",
          "",
          "1. **No future replay** — everything stops at 9 Sep 2026 (your real trade log ends 7 Sep).",
          "2. Each refund buys exactly one new $5K account (fee refund = price of one account).",
          "3. A new account starts trading the signals on/after its buy date (same trades as the "
          "older accounts that day).",
          "4. Split modeled as **90% flat** on every payout.",
          "5. Scaling (every 3 payouts) applies per account: $5K → $7.5K → $10K → $25K → $60K → $150K.",
          "6. Risk per trade = 1% of that account's current level (so a scaled account's dollar "
          "risk grows with its level — that's the growth plan, not you raising risk).",
          ""]
    open(os.path.join(OUT, "multiacct_report.md"), "w").write("\n".join(L))
    print("Report:", os.path.join(OUT, "multiacct_report.md"))


if __name__ == "__main__":
    main()
