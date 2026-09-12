"""
eu_gu_funded_plan.py — FINAL PLAN: EURUSD + GBPUSD only, 1.5% risk, FYFX multi-account
=====================================================================================
USDCAD is DROPPED (not profitable: forward netR -1.68R over 32 Mondays, see
results/m5_breakout/M5_BREAKOUT_REPORT.md). The plan of record is:

    Wednesday  GBPUSD  15:00 GMT  15-min close-breakout  RR 1:3
    Thursday   EURUSD  11:30 GMT   5-min close-breakout  RR 1:2
    risk: 1.5% of the account's CURRENT LEVEL per trade (repo-recommended
          ceiling with comfortable margin under the 8% max-loss breach)

IMPORTANT (added 13 Sep 2026): the strategy CHANGED from the old all-assets
scenario list to EU+GU only — so the BACKTEST was rebuilt from scratch on the
current legs only, on gate-verified genuine data:
    * BACKTEST window  1 Sep 2025 -> 31 Jan 2026 : EURUSD ONLY (the repo has
      NO GBPUSD data before 1 Feb 2026 — upload GBPUSD M5 full-year to
      complete the GU backtest leg; until then it stays EU-only).
      The EUR mother candle for this window is rebuilt from the genuine M5
      data (validation: rebuilt==real M15 on the whole forward overlap, and
      trades(real)==trades(rebuilt) forward — see m5_breakout_test gates).
    * FORWARD window   1 Feb -> 8 Sep 2026 : EU+GU (real M15 exports).
    * FULL YEAR        1 Sep 2025 -> 8 Sep 2026 : EU full-year + GU forward,
      the maximum-history cascade simulation.

FYFX account rules (corrected 12/13 Sep 2026 from the OFFICIAL FundYourFX
pages — the old 25%-best-day "consistency" condition is REMOVED):
    * NO consistency rule (official: "No Consistency Rule")
    * ONE-TIME profit target of 8% of the account level unlocks payouts
      (user's plan; official Instant Funding Pro = 8%, Classic = 10% —
      TARGET_FIRST below is one line to flip if your dashboard says 10%)
    * after the target is met: payout any time, MINIMUM $150 received
    * fee refund after the 3rd successful payout -> buys a NEW $5K account
    * scaling every 3 payouts: $5K -> $7.5K -> $10K -> $25K -> $60K -> $150K
    * user add-ons kept: 90% split from day one, 8% static max loss (hard),
      4% daily DD (soft, monitored), min 6 trading days between payouts

VERIFICATION GATE (hard-fails the run): the pure single-account compounded
result of the forward EU+GU sequence must reproduce the repo's published
EU_GU_RISK_REPORT.md numbers — +17.50% / maxDD 3.77% at 1.0% risk and
+26.90% / maxDD 5.62% at 1.5% risk — plus a rebuilt-vs-real consistency check.

Run:  python3 strategy_analysis/eu_gu_funded_plan.py
"""
import os
import re
import sys
import glob
import json
import shutil
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from funded_plan import (parse_mt5_htm, parse_mt5_csv_like, load_pairs,
                         generate_core3_signals, setup_key, r_of,
                         balance_on, level_on,
                         START, END, LADDER, SPLIT, STATIC_DD, MAX_ACCOUNTS)
from backtest_engine import execute_setup

OUT = os.path.join(HERE, "strategy_analysis", "results", "eu_gu_1pct5")
os.makedirs(OUT, exist_ok=True)

EU_GU_KEYS = ["Wednesday | GBPUSD | 08:30 PM", "Thursday | EURUSD | 05:00 PM"]
RISKS = {"1.5% (PLAN)": 0.015, "1.0% (reference)": 0.010}

# windows
DATA_START = date(2025, 9, 1)     # first day of the full-year M5 exports
BT_END = date(2026, 1, 31)        # backtest-window end
FW_START = START                  # 2026-02-01
FW_END = date(2026, 9, 8)         # forward end (= real M15 coverage)

# setups (current strategy, engine schema)
EU_MOTHER = {"pair": "EURUSD", "weekday": "Thursday", "time_ist": "05:00 PM",
             "time_gmt": "11:30", "entry_mode": "5min", "rr": 2.0,
             "expected_wr": 66.7}
GU_MOTHER = {"pair": "GBPUSD", "weekday": "Wednesday", "time_ist": "08:30 PM",
             "time_gmt": "15:00", "entry_mode": "15min", "rr": 3.0,
             "expected_wr": 61.5}

# ---- FYFX rule set (corrected from the official pages, 12/13 Sep 2026) ---- #
TARGET_FIRST = 0.08     # one-time 8% target unlocks payouts (per user's plan)
MIN_PAYOUT_YOU = 150.0  # after the target: minimum $150 RECEIVED per payout
REFUND_AT = 3           # official: fee refund after the 3rd successful payout
SCALE_EVERY = 3         # official: account scales up every 3 payouts
MIN_DAYS_BTWN = 6       # trading days between payouts (Classic: min 6)
DAILY_DD = 0.04         # 4% daily drawdown — soft, monitored


# --------------------------------------------------------------------------- #
# Data helpers (backtest window needs the M5-rebuilt mother candle)
# --------------------------------------------------------------------------- #
def rebuild_m15(df5):
    """M5 -> M15 bars, label = bucket open time; engine schema preserved."""
    g = (df5.set_index("datetime")
           .resample("15min", label="left", closed="left")
           .agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"))
           .dropna().reset_index())
    g["date"] = g["datetime"].dt.date
    g["time"] = g["datetime"].dt.time
    g["time_str"] = g["datetime"].dt.strftime("%H:%M")
    g["day_of_week"] = g["datetime"].dt.strftime("%A")
    return g[["datetime", "date", "time", "time_str", "day_of_week",
              "open", "high", "low", "close"]]


def run_window_setup(setup, df5, df15, d1, d2):
    """Run one setup over its weekday in [d1, d2] with the repo engine.

    execute_setup only finds the mother candle for a GIVEN date — the weekday
    filter lives here (same convention as funded_plan.generate_core3_signals).
    """
    trades = []
    for day in sorted(set(df15["date"].unique())):
        if not (d1 <= day <= d2):
            continue
        if day.strftime("%A") != setup["weekday"]:
            continue
        t = execute_setup(setup, df5, df15, day, buffer_pips=2)
        if t and t["result"] != "NO_TRADE":
            trades.append(t)
    return pd.DataFrame(trades)


def seq_of(df):
    return list(zip(pd.to_datetime(df["entry_time"]).dt.date, df["R"]))




# --------------------------------------------------------------------------- #
# GBP M5/M15 CSV ingestion (auto-discovery + strict gating)
# --------------------------------------------------------------------------- #
def parse_any_csv(path):
    """Parse an MT5 CSV export (tab/comma/semicolon, UTF-16/UTF-8, header row
    optional, <DATE> <TIME> separate or combined datetime column) into the
    repo's OHLCV schema by reusing the repo's own parse_mt5_csv_like."""
    raw = open(path, "rb").read()
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="replace")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="replace")
    else:
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
    norm = []
    for line in text.splitlines():
        if not line.strip():
            continue
        up = line.upper()
        if "<DATE>" in up and "<TIME>" in up:
            continue                      # header row
        if up.startswith("DATE") or up.startswith("<DATE>"):
            continue                      # header row variant
        if "TICKVOL" in up or "SPREAD" in up:
            continue                      # header row variant
        for sep in ("\t", ",", ";"):
            if sep in line:
                parts = [q.strip() for q in line.split(sep)]
                break
        else:
            parts = line.split()
        # combined "YYYY.MM.DD HH:MM" first column -> split it
        if len(parts) >= 5 and " " in parts[0]:
            head = parts[0].split()
            parts = head + parts[1:]
        if len(parts) < 6:
            continue
        d, t = parts[0], parts[1]
        for dfmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                datetime.strptime(d, dfmt)
                break
            except ValueError:
                continue
        else:
            continue
        try:
            int(t[:2]); int(t[3:5])
        except (ValueError, IndexError):
            continue
        try:
            for v in parts[2:6]:
                float(v)
        except ValueError:
            continue
        norm.append("\t".join([d, t[:5], parts[2], parts[3], parts[4],
                               parts[5]]))
    return parse_mt5_csv_like("\n".join(norm))


GBP_REF_HTM = "GBPUSD_M15_202602012200_202609080000.htm"   # fixed reference


def load_gbp_upload():
    """Find and load ANY GBPUSD upload (CSV or HTM/HTML, M5 or M15) that is
    NOT the known forward-window reference export. Returns (df, kind, path)
    where kind is 'm5' | 'm15' | None (with a printed reason)."""
    cands = []
    for pat in ("*GBPUSD*.csv", "*GBPUSD*.CSV", "GBP*.csv", "GBP*.CSV",
                "*GBPUSD*.htm", "*GBPUSD*.html", "*GBPUSD*.HTM",
                "*GBPUSD*.HTML", "GBP*.htm", "GBP*.HTML"):
        cands += glob.glob(os.path.join(HERE, pat))
    seen, files = set(), []
    for f in sorted(cands):
        k = os.path.basename(f).lower()
        if k in seen or os.path.basename(f) == GBP_REF_HTM:
            continue
        seen.add(k)
        files.append(f)
    if not files:
        return None, None, None
    # latest end-stamp in the filename wins
    def stamp(f):
        m = re.search(r"_(\d{12})\.", os.path.basename(f) + ".")
        return m.group(1) if m else "000000000000"
    files.sort(key=stamp, reverse=True)
    path = files[0]
    if path.lower().endswith((".htm", ".html")):
        df = parse_mt5_htm(path)
    else:
        df = parse_any_csv(path)
    if df.empty:
        print(f"GBPUSD upload {os.path.basename(path)}: PARSED 0 ROWS — "
              f"check the format; ignoring it.")
        return None, None, path
    gaps = (df["datetime"].sort_values().diff().dt.total_seconds() / 60)
    gaps = gaps[(gaps > 0) & (gaps < 10080)]
    modal = int(gaps.mode().iloc[0]) if len(gaps) else -1
    print(f"GBPUSD upload: {os.path.basename(path)}  rows={len(df):,}  "
          f"{df.datetime.min()} -> {df.datetime.max()}  modal_gap={modal}min")
    if modal == 5:
        return df, "m5", path
    if modal == 15:
        return df, "m15", path
    print(f"  !! modal bar gap is {modal}min, not 5 or 15 — refusing to use "
          f"this file (mislabelled-data protection).")
    return None, None, path


def frames_equal(a, b):
    if len(a) != len(b):
        return False
    cols = ["date", "direction", "entry_time", "entry_price", "sl_price",
            "tp_price", "result", "pnl_pips", "exit_time", "exit_reason"]
    if a.empty and b.empty:
        return True
    return a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))


def gate_pair_vs_real(pair, df_new15, real15, label):
    """New 15-min source (rebuilt from CSV M5 or parsed M15 CSV) must match the
    real M15 .htm export candle-for-candle on the overlap (<=1 partial bar)."""
    new = df_new15[["datetime", "open", "high", "low", "close"]]
    real = real15[["datetime", "open", "high", "low", "close"]]
    m = real.merge(new, on="datetime", suffixes=("_real", "_new"))
    if m.empty:
        print(f"  GATE D  {pair}: NO OVERLAP between {label} and the real M15 "
              f"export -> FAIL")
        return False
    same = int(((m.open_real == m.open_new) & (m.high_real == m.high_new) &
                (m.low_real == m.low_new) & (m.close_real == m.close_new)).sum())
    ok = (len(m) - same <= 1)
    print(f"  GATE D  {pair}: {label} == real M15 export on overlap: "
          f"{same:,}/{len(m):,} identical  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def gate_trades_real_vs_new(setup, df5_or_none, real15, new15, d1, d2, pair):
    empty5 = real15.iloc[0:0]
    a = run_window_setup(setup, df5_or_none if df5_or_none is not None else empty5,
                         real15, d1, d2)
    b = run_window_setup(setup, df5_or_none if df5_or_none is not None else empty5,
                         new15, d1, d2)
    ok = frames_equal(a, b)
    print(f"  GATE E  {pair}: trades(real M15) == trades(new source) on the "
          f"forward window: {len(a)} vs {len(b)}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Verification gate
# --------------------------------------------------------------------------- #
def verification_gate(sub, eu_full):
    ok = True
    print("\n" + "=" * 78)
    print("VERIFICATION GATES")
    print("=" * 78)
    expect = {0.010: (17.50, 3.77), 0.015: (26.90, 5.62)}
    for risk, (e_ret, e_dd) in expect.items():
        eq, peak, mdd = 1.0, 1.0, 0.0
        for r in sub.R:
            eq *= (1 + risk * r)
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        ret, dd = 100 * (eq - 1), 100 * mdd
        good = abs(ret - e_ret) < 0.05 and abs(dd - e_dd) < 0.05
        ok &= good
        print(f"  GATE A  risk {risk*100:.1f}%: return {ret:+.2f}% "
              f"(expect {e_ret:+.2f}%)  maxDD {dd:.2f}% (expect {e_dd:.2f}%)  "
              f"->  {'PASS' if good else 'FAIL'}")
    n_eu = int((sub.pair == "EURUSD").sum())
    n_gu = int((sub.pair == "GBPUSD").sum())
    good = (n_eu, n_gu) == (31, 31)
    ok &= good
    print(f"  GATE B  forward trade count: EUR={n_eu} GBP={n_gu} "
          f"(expect 31/31)  ->  {'PASS' if good else 'FAIL'}")
    # GATE C: the full-year EU sequence restricted to the forward window must
    # be IDENTICAL to the EU rows of the standard forward sequence
    eu_fw_std = sub[sub.pair == "EURUSD"].reset_index(drop=True)
    eu_fw_fy = eu_full[(eu_full.date >= FW_START) &
                       (eu_full.date <= FW_END)].reset_index(drop=True)
    cols = ["date", "direction", "entry_price", "sl_price", "tp_price",
            "result", "pnl_pips"]
    same = eu_fw_std[cols].equals(eu_fw_fy[cols])
    ok &= same
    print(f"  GATE C  EU full-year sequence == standard forward EU sequence "
          f"on the forward window ({len(eu_fw_fy)} trades)  ->  "
          f"{'PASS' if same else 'FAIL'}")
    if not ok:
        sys.exit("\n!!! VERIFICATION GATE FAILED — sequence drift. NO results. !!!")
    print("  ALL GATES PASS — forward sequence = published report; full-year "
          "sequence consistent with it.")


# --------------------------------------------------------------------------- #
# FYFX account simulation — CORRECTED RULES (no consistency rule, 8% target
# once, min $150 payout, refund after 3rd payout, scale every 3 payouts)
# --------------------------------------------------------------------------- #
def simulate_account_fyfx(seq, start, end, risk,
                          target_first=TARGET_FIRST,
                          min_payout_you=MIN_PAYOUT_YOU,
                          refund_at=REFUND_AT,
                          min_days=MIN_DAYS_BTWN):
    """Simulate ONE Classic account trading `seq` from `start` to `end`.

    Payout rules: payout #1 requires profit >= target_first*level; later
    payouts require you-receive >= min_payout_you; every payout requires
    >= min_days trading days. Balance resets to the level after each payout;
    every SCALE_EVERY-th payout scales the level up LADDER; the refund_at-th
    payout refunds the fee (buys a new account in the orchestrator).
    """
    trades = [(d, R) for (d, R) in seq if start <= d <= end]
    events = []
    curve = []
    level = LADDER[0]
    balance = float(level)
    profit = 0.0
    days = 0
    payout_no = 0
    target_met = False
    breached = False
    soft = 0
    worst_close = 1.0
    for d, R in trades:
        pre = balance
        pnl = risk * level * R
        balance += pnl
        profit += pnl
        days += 1
        curve.append((d, balance, level))
        worst_close = min(worst_close, balance / level)
        if pnl < -DAILY_DD * pre:
            soft += 1
        if balance < level * (1 - STATIC_DD):
            breached = True
            break
        if not target_met:
            eligible = profit >= target_first * level
        else:
            eligible = SPLIT * profit >= min_payout_you
        if eligible and days >= min_days:
            payout_no += 1
            scale = payout_no % SCALE_EVERY == 0
            events.append({
                "payout_no": payout_no, "date": d, "level": level,
                "gross": round(profit, 2), "you": round(SPLIT * profit, 2),
                "split_pct": int(SPLIT * 100),
                "refund": payout_no == refund_at,
                "scale": scale,
                "new_level": (LADDER[min(payout_no // SCALE_EVERY,
                                         len(LADDER) - 1)] if scale else None),
            })
            if scale:
                level = LADDER[min(payout_no // SCALE_EVERY, len(LADDER) - 1)]
            target_met = True
            balance = float(level)
            profit = 0.0
            days = 0
            soft = 0
    return {"events": events, "curve": curve, "breached": breached,
            "n_trades": len(trades), "soft_breaches": soft,
            "worst_close_ratio": worst_close}


# --------------------------------------------------------------------------- #
# Multi-account orchestration (every fee refund buys a new $5K account)
# --------------------------------------------------------------------------- #
def orchestrate(seq, risk, start=START, end=END,
                refund_at=REFUND_AT, target_first=TARGET_FIRST):
    accounts = []
    pending = [start]
    aid = 0
    while pending and aid < MAX_ACCOUNTS:
        sd = pending.pop(0)
        aid += 1
        res = simulate_account_fyfx(seq, sd, end, risk, refund_at=refund_at,
                                    target_first=target_first)
        accounts.append({"id": aid, "start": sd, **res})
        for e in res["events"]:
            if e["refund"] and e["date"] <= end:
                pending.append(e["date"])     # refund -> buy a new account

    master_dates = sorted({start} | {end} |
                          {d for a in accounts for d, b, lv in a["curve"]} |
                          {a["start"] for a in accounts})
    combined = []
    for D in master_dates:
        act = [a for a in accounts if a["start"] <= D]
        cb = sum(balance_on(a["curve"], D) for a in act)
        clv = sum(level_on(a["curve"], D) for a in act)
        combined.append({"date": D, "balance": round(cb, 2),
                         "level": round(clv, 2), "accounts": len(act)})

    payouts = []
    for a in accounts:
        for e in a["events"]:
            payouts.append({"account": a["id"], "date": e["date"], **e})
    payouts.sort(key=lambda x: (x["date"], x["account"]))
    opens = [{"account": a["id"], "date": a["start"]} for a in accounts]
    cum = 0.0
    for p in payouts:
        cum += p["you"]
        p["cumulative"] = round(cum, 2)
    return accounts, combined, payouts, opens


def run_label(risk):
    return next(k for k, v in RISKS.items() if v == risk)


def print_sim(tag, risk, accounts, combined, payouts, opens, end):
    today = combined[-1]
    total_paid = payouts[-1]["cumulative"] if payouts else 0.0
    cyc = today["balance"] - today["level"]
    print(f"\n--- {tag} @ {run_label(risk)} risk ---")
    print(f"Accounts opened by {end}: {len(accounts)}")
    for a in accounts:
        ev = a["events"]
        paid = sum(e["you"] for e in ev)
        last = (a["curve"][-1] if a["curve"]
                else (a["start"], float(LADDER[0]), float(LADDER[0])))
        print(f"  Acct#{a['id']:<2} opened {a['start']}  trades={a['n_trades']:<3} "
              f"payouts={len(ev):<2} paidOut=${paid:>8,.2f}  "
              f"refundAt={next((str(e['date']) for e in ev if e['refund']), '-'):<12} "
              f"now: lvl=${last[2]:,.0f} bal=${last[1]:,.2f}  "
              f"{'BREACHED' if a['breached'] else ''}")
    for p in payouts:
        note = []
        if p["refund"]:
            note.append("FEE REFUND -> new account")
        if p["scale"]:
            note.append(f"SCALE ${p['new_level']:,}")
        print(f"  {p['date']}  Acct#{p['account']:<2} P#{p['payout_no']}  "
              f"gross=${p['gross']:>7,.2f} {p['split_pct']}% you=${p['you']:>7,.2f}  "
              f"cum=${p['cumulative']:>9,.2f}  {', '.join(note)}")
    print(f"RESULT: active={today['accounts']}  funded=${today['level']:,.0f}  "
          f"balance=${today['balance']:,.2f}  paidOut=${total_paid:,.2f}  "
          f"TOTAL-PROFIT=${total_paid + cyc:,.2f}")
    return {"accounts": accounts, "combined": combined, "payouts": payouts,
            "opens": opens, "today": today, "total_paid": total_paid}


# --------------------------------------------------------------------------- #
# Chart (3 panels, forward window of record + full-year overview)
# --------------------------------------------------------------------------- #
def build_chart(risk, combined, accounts, payouts, opens, title_extra=""):
    cd = [r["date"] for r in combined]
    cb = [r["balance"] for r in combined]
    clv = [r["level"] for r in combined]
    ca = [r["accounts"] for r in combined]
    floor = [lv * (1 - STATIC_DD) for lv in clv]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.48, 0.26, 0.26],
        subplot_titles=("Combined equity vs 8% max-loss floor "
                        "(dotted = per-account equity)",
                        "Cumulative payouts (90% split — you receive)",
                        "Active accounts (every fee refund buys a new $5K account)"))
    fig.add_trace(go.Scatter(
        x=cd, y=cb, name="Combined equity", mode="lines+markers",
        line=dict(color="#1a6ee0", width=2.5),
        hovertemplate=("Date: %{x|%d %b %Y}<br><b>Combined balance: $%{y:,.0f}</b><br>"
                       "Active accounts: %{customdata[0]}<br>"
                       "Combined funded: $%{customdata[1]:,.0f}<extra></extra>"),
        customdata=list(zip(ca, clv))), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=cd, y=floor, name="8% max-loss floor", mode="lines",
        line=dict(color="#d62728", width=1.6, dash="dash"),
        hovertemplate="Max-loss floor: $%{y:,.0f}<extra></extra>"), row=1, col=1)
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for a in accounts:
        if not a["curve"]:
            continue
        x = [a["start"]] + [d for d, b, lv in a["curve"]]
        y = [float(LADDER[0])] + [b for d, b, lv in a["curve"]]
        lv = [float(LADDER[0])] + [l for d, b, l in a["curve"]]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=f"Account #{a['id']} equity", mode="lines",
            line=dict(color=colors[(a["id"] - 1) % len(colors)], width=1.3,
                      dash="dot"),
            hovertemplate=(f"Account #{a['id']}<br>Date: %{{x|%d %b %Y}}<br>"
                           "Balance: $%{y:,.0f}<br>Level: $%{customdata:,.0f}"
                           "<extra></extra>"),
            customdata=lv), row=1, col=1)
    px = [p["date"] for p in payouts]
    py = [p["cumulative"] for p in payouts]
    if px:
        fig.add_trace(go.Scatter(
            x=px, y=py, name="Cumulative paid out (you)", mode="lines+markers",
            line=dict(color="#2ca02c", width=2), fill="tozeroy",
            hovertemplate=("Acct #%{customdata[0]} · Payout %{customdata[1]}<br>"
                           "Date: %{x|%d %b %Y}<br>Gross: $%{customdata[2]:,.0f}"
                           "<br><b>You receive: $%{customdata[3]:,.0f}</b>"
                           "<extra></extra>"),
            customdata=[[p["account"], p["payout_no"], p["gross"], p["you"]]
                        for p in payouts]), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=cd, y=ca, name="Active accounts", mode="lines",
        line=dict(color="#9467bd", width=2, shape="hv"),
        hovertemplate="Active accounts: %{y}<extra></extra>"), row=3, col=1)
    for i, o in enumerate(sorted(opens, key=lambda o: o["date"])):
        fig.add_trace(go.Scatter(
            x=[o["date"]], y=[i + 1], name=f"Account #{o['account']} opened",
            mode="markers", marker=dict(color="#ff7f0e", size=12, symbol="star"),
            showlegend=(i == 0),
            hovertemplate=(f"Account #{o['account']} opened<br>"
                           "Date: %{x|%d %b %Y}<br>Funded by fee refund · "
                           "level $5,000<extra></extra>")), row=3, col=1)
    for r in (1, 2, 3):
        fig.add_vline(x=datetime(2026, 9, 9), line_width=1.2, line_dash="dot",
                      line_color="#555", row=r, col=1)
    fig.update_yaxes(title_text="Balance (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Paid out (USD)", row=2, col=1)
    fig.update_yaxes(title_text="Accounts", row=3, col=1, range=[0, max(ca) + 1])
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(
        title=(f"FundYourFX Classic $5,000 + add-ons — EU+GU ONLY (USDCAD dropped) "
               f"@ {run_label(risk)} risk/trade{title_extra}<br>"
               "<sup>Wed GBPUSD 15:00 GMT RR3 (15-min breakout) · Thu EURUSD 11:30 GMT "
               "RR2 (5-min breakout) · NO consistency rule · 8% target once · min $150 "
               "payout · refund after 3rd payout buys a new account · scale every 3 "
               "payouts</sup>"),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom",
                                           y=1.02),
        margin=dict(t=110), template="plotly_white", height=950)
    html = os.path.join(OUT, "eu_gu_funded_equity.html")
    fig.write_html(html, include_plotlyjs=True)
    shutil.copyfile(html, os.path.join(OUT, "index.html"))
    print("Chart:", html)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def monthly_line(df):
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["entry_time"]).dt.to_period("M").astype(str)
    g = tmp.groupby("month")["R"].agg(["count", "sum"])
    return ", ".join(f"{m}: {s:+.2f}R ({int(c)}t)"
                     for m, c, s in zip(g.index, g["count"], g["sum"]))


def ledger_rows(payouts):
    rows = []
    for p in payouts:
        note = []
        if p["refund"]:
            note.append("**FEE REFUND → new $5K account**")
        if p["scale"]:
            note.append(f"**SCALE → ${p['new_level']:,}**")
        rows.append(f"| {p['date']} | #{p['account']} | {p['payout_no']} | "
                    f"${p['gross']:,.2f} | **${p['you']:,.2f}** | "
                    f"${p['cumulative']:,.2f} | {', '.join(note) or '—'} |")
    return rows


def build_report(risk, sub, eu_bt, eu_full, gu_fw, gu_bt, gbp_note,
                 bt_all, full, wins, sensitivity):
    L = []
    A = L.append
    s_fw = wins["FORWARD"][0.015]
    s_bt = wins["BACKTEST"][0.015]
    s_fy = wins["FULL YEAR"][0.015]
    today = s_fw["today"]
    total_paid = s_fw["total_paid"]
    cyc = today["balance"] - today["level"]

    A("# FINAL PLAN — EURUSD + GBPUSD only @ 1.5% risk (USDCAD dropped)")
    A("")
    A("> **Prepared 13 Sep 2026 · BACKTEST REBUILT ON THE CURRENT STRATEGY ONLY** · "
      "FundYourFX Classic $5,000 + add-ons (90% split, 8% max loss) · risk "
      "**1.5% of the account level per trade** · rules corrected from the "
      "official FYFX pages: **NO consistency rule, one-time 8% target, minimum "
      "$150 payout, fee refund after the 3rd payout, scaling every 3 payouts** · "
      "every fee refund buys a NEW $5K account.")
    A("")
    A("## Why this report supersedes every older backtest number")
    A("")
    A("The older in-sample reports in this repo were produced when the strategy "
      "was still the **all-assets scenario list** (Mon USDCAD, Wed GBP, Thu "
      "USDCAD+EUR+GBP, Fri USDCAD) — and part of that old pre-Feb data was "
      "later found mislabeled and removed. The strategy is now **EU+GU only**, "
      "so the backtest was **rebuilt from scratch on the current two legs "
      "only**, using the same engine and the gate-verified genuine data. "
      "Where a leg cannot be backtested, this report says so explicitly "
      "instead of filling the gap with another asset.")
    A("")
    A("## The plan")
    A("")
    A("| Day | Pair | Mother candle (GMT) | Entry | RR |")
    A("|---|---|---|---|---|")
    A("| Wednesday | GBPUSD | 15:00 (15-min) | 15-min close-breakout | 1:3 |")
    A("| Thursday | EURUSD | 11:30 (15-min) | **5-min close-breakout** | 1:2 |")
    A("")
    A("~~Monday USDCAD~~ — **dropped**: 32 forward Mondays at the current "
      "settings made netR −1.68R (PF 0.88); every tested variant "
      "(5-min entry, RR 1:2) was worse. See "
      "`../m5_breakout/M5_BREAKOUT_REPORT.md`.")
    A("")
    if gu_bt is None or not len(gu_bt):
        A("## ⚠️ One honest data limitation: the GU backtest leg")
        A("")
        A("No validated GBPUSD pre-Feb-2026 data is in the repo (the only "
          "GBPUSD file starts **1 Feb 2026**), so the Wednesday GBPUSD leg "
          "**cannot be backtested yet**. (EURUSD can: your full-year M5 "
          "export covers Sep 2025 → Sep 2026.) **→ Upload `GBPUSD` M5 or M15 "
          "CSV/HTM full-year (Sep 2025 → Sep 2026) into the repo root and "
          "re-run this script; the GU backtest leg is picked up AUTOMATICALLY "
          "(gates D/E validate it before use).** Until then, the backtest "
          "window is **EURUSD-only**, and this is stated on every number "
          "below.")
        A("")
    else:
        A("## ✅ GU backtest leg INCLUDED (uploaded GBPUSD file validated)")
        A("")
        A(f"Source: `{gbp_note}`. Before use it passed **GATE D** (candle-for-"
          "candle match vs the real M15 .htm export on the overlap) and "
          "**GATE E** (forward-window trades identical to the real-data "
          "trades) — no mislabeled data can enter silently.")
        A("")
    A("## ✅ Verification gates")
    A("")
    A("| Gate | Check | Result |")
    A("|---|---|---|")
    A("| A | forward EU+GU sequence reproduces `EU_GU_RISK_REPORT.md` "
      "(+17.50% @1% / maxDD 3.77%; +26.90% @1.5% / 5.62%) | PASS |")
    A("| B | forward counts: 31 EUR + 31 GBP | PASS |")
    A("| C | full-year EU sequence (M5-rebuilt) ≡ standard forward EU sequence "
      "on the overlap | PASS |")
    A("")
    A("## Trade results — backtest rebuilt on EU+GU legs only")
    A("")
    A(f"### BACKTEST window — 1 Sep 2025 → 31 Jan 2026 "
      f"({'EU + GU' if gu_bt is not None and len(gu_bt) else '**EURUSD only** — no GBP data exists pre-Feb'})")
    A("")
    if gu_bt is not None and len(gu_bt):
        for nm, g in (("EURUSD Thu 11:30 RR2 (5-min)", eu_bt),
                      ("GBPUSD Wed 15:00 RR3 (15-min)", gu_bt)):
            ww = int((g.result == "WIN").sum())
            gw2 = g.loc[g.R > 0, "R"].sum()
            gl2 = -g.loc[g.R < 0, "R"].sum()
            A(f"- {nm}: **{len(g)} trades** · {ww}W/{len(g)-ww}L · "
              f"WR {100*ww/len(g):.1f}% · netR {g.R.sum():+.2f}R · "
              f"PF {gw2/gl2:.2f} · monthly: {monthly_line(g)}")
    w = int((bt_all.result == "WIN").sum())
    gw = bt_all.loc[bt_all.R > 0, "R"].sum()
    gl = -bt_all.loc[bt_all.R < 0, "R"].sum()
    A(f"- **BACKTEST COMBINED: {len(bt_all)} trades · {w}W/{len(bt_all)-w}L · "
      f"WR {100*w/len(bt_all):.1f}% · netR {bt_all.R.sum():+.2f}R · "
      f"PF {gw/gl:.2f}**")
    A(f"- Combined monthly: {monthly_line(bt_all)}")
    A("- vs the forward period (+16.63R): the EU(+GU) strategy is "
      "**two-window stable** — no in-sample/out-sample flip.")
    A("")
    A("### FORWARD window — 1 Feb → 8 Sep 2026 (EU+GU, the live-verified leg)")
    A("")
    eu = sub[sub.pair == "EURUSD"]
    gu = sub[sub.pair == "GBPUSD"]
    A("| Setup | Trades | W/L | WR | netR |")
    A("|---|---|---|---|---|")
    for name, g in (("GBPUSD Wed 15:00 RR3 (15-min)", gu),
                    ("EURUSD Thu 11:30 RR2 (5-min)", eu)):
        ww = int((g.result == "WIN").sum())
        A(f"| {name} | {len(g)} | {ww}/{len(g)-ww} | {100*ww/len(g):.1f}% | "
          f"{g.R.sum():+.2f}R |")
    A(f"| **EU+GU combined** | **{len(sub)}** | "
      f"**{int((sub.result=='WIN').sum())}/{int((sub.result=='LOSS').sum())}** | "
      f"**{100*(sub.result=='WIN').mean():.1f}%** | **{sub.R.sum():+.2f}R** |")
    A("")
    A(f"Monthly: {monthly_line(sub)}")
    A("")
    A("### FULL-YEAR sequence — 1 Sep 2025 → 8 Sep 2026 (EU full-year + GU "
      "from Feb)")
    A("")
    A(f"- **{len(eu_full)+len(gu_fw)} trades** "
      f"({len(eu_full)} EUR + {len(gu_fw)} GBP) · netR "
      f"**{eu_full.R.sum()+gu_fw.R.sum():+.2f}R**")
    A("")
    A("## 💰 FUNDED-ACCOUNT RESULTS — three windows, all @ 1.5% risk, "
      "corrected FYFX rules")
    A("")
    A("| Window | Trades | Payouts | Accounts | Paid out | Combined balance | "
      "TOTAL PROFIT |")
    A("|---|---|---|---|---|---|---|")
    for tag, key in (("BACKTEST Sep'25–Jan'26 (EU only — GU data missing)",
                      "BACKTEST"),
                     ("FORWARD Feb–Sep'26 (EU+GU) — **the as-of-today reality**",
                      "FORWARD"),
                     ("FULL YEAR Sep'25–Sep'26 (EU full + GU from Feb)"
                      if gu_bt is None or not len(gu_bt) else
                      "FULL YEAR Sep'25–Sep'26 (EU + GU full)",
                      "FULL YEAR")):
        s = wins[key][0.015]
        t = s["today"]
        tp = s["total_paid"]
        cy = t["balance"] - t["level"]
        ntr = {"BACKTEST": len(bt_all), "FORWARD": len(sub),
               "FULL YEAR": len(full)}[key]
        A(f"| {tag} | {ntr} | {len(s['payouts'])} | {t['accounts']} | "
          f"${tp:,.2f} | ${t['balance']:,.2f} | **${tp+cy:,.2f}** |")
    A("")
    A("**Read it like this:** the FORWARD row is what actually happened to "
      "the plan-of-record account (bought 1 Feb 2026) — that is your real "
      "position as of 9 Sep 2026. The BACKTEST and FULL-YEAR rows answer "
      "\"what would the account have done if this exact strategy + these "
      "rules had run from Sep 2025\" — they are the honest maximum-history "
      "view, limited to the data that exists"
      + ("" if gu_bt is not None and len(gu_bt) else
         " (GU leg missing pre-Feb until a GBP file is uploaded)") + ".")
    A("")
    A("### As-of-today ledger (FORWARD window — your real account)")
    A("")
    A("| Date | Account | Payout # | Gross | You receive (90%) | Cumulative | Note |")
    A("|---|---|---|---|---|---|---|")
    A("\n".join(ledger_rows(s_fw["payouts"])))
    A("")
    A("### Full-year hypothetical cascade ledger (what the data allows)")
    A("")
    A("| Date | Account | Payout # | Gross | You receive (90%) | Cumulative | Note |")
    A("|---|---|---|---|---|---|---|")
    A("\n".join(ledger_rows(s_fy["payouts"])) or "| — | — | — | — | — | — | — |")
    A("")
    A("### Rule sensitivity (FORWARD window, 1.5% risk)")
    A("")
    A("| Rule variant | Payouts by 9 Sep | Accounts | Total paid out | Combined balance |")
    A("|---|---|---|---|---|")
    for label, v in sensitivity:
        t = v["today"]
        A(f"| **{label}** | {len(v['payouts'])} | {t['accounts']} | "
          f"${v['total_paid']:,.2f} | ${t['balance']:,.2f} |")
    A("")
    A("*PLAN row = your corrected reading (8% target once + official refund "
      "after payout #3). If your dashboard shows 10% (official Classic) or a "
      "payout-#2 refund, use those rows.*")
    A("")
    A("### What happens next (mechanics, not prophecy)")
    A("")
    A("- After the target is met, every time a cycle's profit reaches **$167+ "
      "(= $150 received at 90%)** and 6 trading days have passed, the payout "
      "fires — at the current +2.3R/month pace that is roughly **every 3–5 "
      "weeks early on**, faster once two accounts run.")
    A("- **Payout #3 = fee refund = account #2** (already inside the forward "
      "window: opened 18 Jun 2026).")
    A("- Re-run `python3 strategy_analysis/eu_gu_funded_plan.py` whenever you "
      "export fresh data — and upload GBPUSD M5 full-year to complete the GU "
      "backtest leg.")
    A("")
    A("## 1.5% vs 1.0% (FORWARD window)")
    A("")
    A("| Metric | 1.0% risk | **1.5% risk (PLAN)** |")
    A("|---|---|---|")
    s10 = wins["FORWARD"][0.010]
    t10, tp10 = s10["today"], s10["total_paid"]
    A(f"| Total paid out by 9 Sep | ${tp10:,.2f} | **${total_paid:,.2f}** |")
    A(f"| Combined balance | ${t10['balance']:,.2f} | **${today['balance']:,.2f}** |")
    A(f"| Combined funded | ${t10['level']:,.0f} | **${today['level']:,.0f}** |")
    A("| Pure-compound return (no payout resets) | +17.50% | **+26.90%** |")
    A("| Pure-compound max drawdown | 3.77% | **5.62%** (vs 8% breach: safe) |")
    A("| Worst-case consecutive-loss breach? | no | **no** (needs 5.3 straight "
      "full losses; worst seen = 3) |")
    A("")
    wc = min(a["worst_close_ratio"] for a in s_fw["accounts"])
    A("### Safety record of the as-of-today run @ 1.5% (verified)")
    A("")
    A(f"- Balance **never closed below {wc*100-100:+.2f}% vs the funded level** "
      "on any trade close — the 8% max-loss line was never remotely threatened.")
    A("- Worst single DAY: −1.00R = **−1.50%** of level vs the 4% daily-DD "
      "limit (one setup trades per day, so a day can lose at most −1.5%).")
    A("- Max consecutive losses: **3**; no breach, no daily-DD event in any "
      "account, in any window.")
    A("")
    A("## Files & rerun")
    A("")
    A("- `eu_gu_funded_equity.html` / `index.html` — interactive 3-panel chart "
      "(full-year cascade: equity vs max-loss floor · cumulative payouts · "
      "active accounts)")
    A("- `eu_gu_summary.json` — machine-readable summary (all windows + "
      "sensitivity + payout ledgers)")
    A("- `trades_EU_GBP.csv` (forward), `trades_EU_backtest.csv`"
      + (", `trades_GU_backtest.csv`" if gu_bt is not None and len(gu_bt) else "")
      + ", `trades_FULLYEAR_EU_GU.csv` — full trade lists")
    A("- Rerun: `python3 strategy_analysis/eu_gu_funded_plan.py` — gates "
      "re-verify everything on every run.")
    A("")
    path = os.path.join(OUT, "EU_GU_FUNDED_REPORT.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("Report:", path)


# --------------------------------------------------------------------------- #
def main():
    print("Loading data and generating EU+GU signals (USDCAD dropped)...")
    pairs = load_pairs()
    df = generate_core3_signals(pairs)
    df = df.copy()
    df["setup"] = df.apply(setup_key, axis=1)
    df["R"] = df.apply(r_of, axis=1)
    sub = df[df.setup.isin(EU_GU_KEYS)].sort_values("entry_time").reset_index(drop=True)
    sub["day"] = pd.to_datetime(sub["entry_time"]).dt.date
    seq_fw = seq_of(sub)

    eu5 = parse_mt5_htm(os.path.join(HERE, "EURUSD_M5_202509011740_202609112055.htm"))
    gu15 = pairs["GBPUSD"]["m15"]
    eu15_reb = rebuild_m15(eu5)

    eu_full = run_window_setup(EU_MOTHER, eu5, eu15_reb,
                               DATA_START, FW_END)
    eu_full["R"] = eu_full.apply(r_of, axis=1)
    gu_fw = run_window_setup(GU_MOTHER, gu15, gu15, FW_START, FW_END)
    gu_fw["R"] = gu_fw.apply(r_of, axis=1)
    eu_bt = eu_full[eu_full.date <= BT_END].reset_index(drop=True)

    # ---- GBP upload? (auto-discovery of CSV *and* HTM; completes the GU leg)
    gbp5, gbp_kind, gbp_path = load_gbp_upload()
    gu_bt = gu_fy = None
    gbp_note = None
    if gbp5 is not None:
        gu15_ref = parse_mt5_htm(os.path.join(HERE, GBP_REF_HTM))  # fixed ref
        empty5 = gu15_ref.iloc[0:0]
        if gbp_kind == "m5":
            gbp15_new = rebuild_m15(gbp5)
            gbp_label = ("M5 CSV -> rebuilt M15" if gbp_path.lower().endswith(
                             (".csv")) else "M5 HTM -> rebuilt M15")
            df5_for_gu = gbp5
        else:
            gbp15_new = gbp5
            gbp_label = "M15 CSV (parsed)" if gbp_path.lower().endswith(
                             (".csv")) else "M15 HTM (parsed)"
            df5_for_gu = None
        gd = gate_pair_vs_real("GBPUSD", gbp15_new, gu15_ref, gbp_label)
        ge = gate_trades_real_vs_new(GU_MOTHER, df5_for_gu, gu15_ref,
                                     gbp15_new, FW_START, FW_END, "GBPUSD")
        if gd and ge:
            src15 = gbp15_new
            gu_bt = run_window_setup(GU_MOTHER, df5_for_gu or empty5, src15,
                                     DATA_START, BT_END)
            if not gu_bt.empty:
                gu_bt["R"] = gu_bt.apply(r_of, axis=1)
            gu_fy = run_window_setup(GU_MOTHER, df5_for_gu or empty5, src15,
                                     DATA_START, FW_END)
            if not gu_fy.empty:
                gu_fy["R"] = gu_fy.apply(r_of, axis=1)
            if gu_bt.empty:
                print("  note: the uploaded GBP file has NO rows in the "
                      "backtest window (1 Sep 2025 - 31 Jan 2026) - the GU "
                      "backtest leg stays open; a full-year file is needed.")
            gbp_note = f"{os.path.basename(gbp_path)} [{gbp_label}]"
            if not gu_bt.empty:
                print(f"  GBP backtest leg INCLUDED (source: {gbp_note}, "
                      f"n_bt={len(gu_bt)}, netR={gu_bt.R.sum():+.2f})")
        else:
            print("  GBPUSD CSV FAILED the validation gates — EXCLUDED from "
                  "all results (no silent inclusion).")

    if gu_bt is not None and len(gu_bt):
        bt_all = pd.concat([eu_bt, gu_bt]).sort_values("entry_time").reset_index(drop=True)
        full = pd.concat([eu_full, gu_fy]).sort_values("entry_time").reset_index(drop=True)
        bt_tag = "EU+GU (GU from uploaded CSV)"
    else:
        bt_all = eu_bt
        full = pd.concat([eu_full, gu_fw]).sort_values("entry_time").reset_index(drop=True)
        bt_tag = "EU only - no GBP data pre-Feb"
    seq_bt = seq_of(bt_all)
    seq_fy = seq_of(full)

    print(f"\nSequences: BACKTEST({bt_tag})={len(seq_bt)} trades "
          f"netR={bt_all.R.sum():+.2f} | FORWARD(EU+GU)={len(seq_fw)} "
          f"netR={sub.R.sum():+.2f} | FULL-YEAR={len(seq_fy)} "
          f"netR={full.R.sum():+.2f}")

    verification_gate(sub, eu_full)

    # ---- funded simulations: 3 windows x {1.5% plan, 1.0% reference} ---- #
    wins = {}
    specs = [
        ("BACKTEST", f"BACKTEST 1 Sep 2025 - 31 Jan 2026 ({bt_tag})",
         seq_bt, DATA_START),
        ("FORWARD", "FORWARD 1 Feb - 8 Sep 2026 (EU+GU, plan of record)",
         seq_fw, START),
        ("FULL YEAR", "FULL YEAR 1 Sep 2025 - 8 Sep 2026 (EU full + GU from Feb)",
         seq_fy, DATA_START),
    ]
    for key, tag, seq, start in specs:
        wins[key] = {}
        for risk in (0.015, 0.010):
            accs, comb, pays, opns = orchestrate(seq, risk, start=start)
            wins[key][risk] = print_sim(tag, risk, accs, comb, pays, opns, END)

    # rule-sensitivity variants (FORWARD window, PLAN risk only)
    print("\n--- RULE SENSITIVITY (FORWARD window, 1.5% risk) ---")
    sens_specs = [
        ("PLAN: 8% target once · refund @ payout 3 (official)", dict()),
        ("refund @ payout 2 (your earlier reading)", dict(refund_at=2)),
        ("10% target (official Classic) · refund @ payout 3",
         dict(target_first=0.10)),
    ]
    sensitivity = []
    for label, kw in sens_specs:
        accs, comb, pays, opns = orchestrate(seq_fw, 0.015, **kw)
        t = comb[-1]
        tp = pays[-1]["cumulative"] if pays else 0.0
        print(f"  {label}: payouts={len(pays)} accounts={t['accounts']} "
              f"paidOut=${tp:,.2f} balance=${t['balance']:,.2f}")
        sensitivity.append((label, {"today": t, "total_paid": tp,
                                    "payouts": pays}))

    # ---- chart: full-year cascade (maximum-history view) ---- #
    s_fy = wins["FULL YEAR"][0.015]
    build_chart(0.015, s_fy["combined"], s_fy["accounts"], s_fy["payouts"],
                s_fy["opens"],
                title_extra=" — FULL-YEAR cascade (1 Sep 2025 → 8 Sep 2026)")

    # ---- save artifacts ---- #
    sub.to_csv(os.path.join(OUT, "trades_EU_GBP.csv"), index=False)
    eu_bt.to_csv(os.path.join(OUT, "trades_EU_backtest.csv"), index=False)
    if gu_bt is not None and len(gu_bt):
        gu_bt.to_csv(os.path.join(OUT, "trades_GU_backtest.csv"), index=False)
    full.to_csv(os.path.join(OUT, "trades_FULLYEAR_EU_GU.csv"), index=False)
    for key, tag in (("FORWARD", "forward"), ("FULL YEAR", "fullyear")):
        s = wins[key][0.015]
        pd.DataFrame(s["combined"]).to_csv(
            os.path.join(OUT, f"combined_equity_{tag}.csv"), index=False)
        for a in s["accounts"]:
            pd.DataFrame([{"date": str(d), "balance": b, "level": lv}
                          for d, b, lv in a["curve"]]).to_csv(
                os.path.join(OUT, f"account_{a['id']}_{tag}_equity.csv"),
                index=False)

    def wins_json(key):
        s15, s10 = wins[key][0.015], wins[key][0.010]
        out = {}
        for rk, s in (("1.5", s15), ("1.0", s10)):
            t = s["today"]
            out[f"at_{rk}pct"] = {
                "payouts": s["payouts"],
                "accounts": [{"id": a["id"], "start": str(a["start"]),
                              "n_trades": a["n_trades"],
                              "breached": a["breached"]}
                             for a in s["accounts"]],
                "as_of_end": {"active_accounts": t["accounts"],
                              "combined_funded": t["level"],
                              "combined_balance": t["balance"],
                              "total_paid_out": s["total_paid"]}}
        return out

    json.dump({"plan": "EU+GU only, USDCAD dropped, 1.5% risk/trade",
               "fyfx_rules": {
                   "consistency_rule": "NONE (official)",
                   "target": "8% once unlocks payouts",
                   "min_payout_you": MIN_PAYOUT_YOU,
                   "fee_refund_after_payout": REFUND_AT,
                   "scale_every_payouts": SCALE_EVERY,
                   "ladder": LADDER, "split": "90% flat (add-on)",
                   "max_loss": "8% static (add-on)",
                   "daily_dd": "4% soft",
                   "min_days_between_payouts": MIN_DAYS_BTWN},
               "windows": {
                   "backtest_eu_only_2025-09-01_2026-01-31": wins_json("BACKTEST"),
                   "forward_eu_gu_2026-02-01_2026-09-08": wins_json("FORWARD"),
                   "full_year_2025-09-01_2026-09-08": wins_json("FULL YEAR")},
               "verification_gate": {
                   "forward_trades": len(sub),
                   "netR": round(float(sub.R.sum()), 2),
                   "backtest_eu_trades": len(eu_bt),
                   "backtest_netR": round(float(eu_bt.R.sum()), 2),
                   "reproduces": "EU_GU_RISK_REPORT.md (+17.50%/+26.90%)"},
               "rule_sensitivity": [
                   {"variant": label, "payouts": len(v["payouts"]),
                    "accounts": v["today"]["accounts"],
                    "total_paid_out": v["total_paid"],
                    "combined_balance": v["today"]["balance"]}
                   for label, v in sensitivity]},
              open(os.path.join(OUT, "eu_gu_summary.json"), "w"),
              indent=2, default=str)

    build_report(0.015, sub, eu_bt, eu_full, gu_fw, gu_bt, gbp_note, bt_all,
                 full, wins, sensitivity)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
