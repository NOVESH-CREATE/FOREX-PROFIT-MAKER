"""
funded_plan.py — THE 3-SETUP FUNDED PLAN (single self-contained script)
========================================================================

Final trading plan, modelled end-to-end on a FundYourFX Classic $5,000 account
with add-ons (90% split from day one, 8% max loss):

  Strategy : core-3 ORB setups, 5-minute/15-minute breakouts, 1% risk per trade
               Mon  USDCAD 15:45 GMT  RR 1:3   (15-min ORB breakout)
               Wed  GBPUSD 15:00 GMT  RR 1:3   (15-min ORB breakout)
               Thu  EURUSD 11:30 GMT  RR 1:2   (5-min ORB breakout)
  Data     : the 4 MT5 .htm exports in the repo root (EURUSD M15/M5,
             GBPUSD M15, USDCAD M15 — forward window 1 Feb -> 8 Sep 2026)
  Accounts : Classic $5K + add-ons: 10% profit target, 8% static max loss
             (hard breach), 4% daily drawdown (soft), min 6 trading days,
             25% best-day payout rule, 90% profit split, fee refund after the
             2nd payout of each account, per-account scaling every 3 payouts:
             $5K -> $7.5K -> $10K -> $25K -> $60K -> $150K
  Growth   : multi-account "reinvest fee refunds" model — every fee refund
             (after an account's 2nd payout) buys a NEW $5K account that starts
             trading the SAME core-3 signals from that date. Risk stays 1% of
             the account's level (never increased).
  Window   : 1 Feb 2026 -> 9 Sep 2026 ONLY (no future replay).

Run (from the repo root, with pandas / numpy / plotly installed):

    python3 funded_plan.py

Expected final numbers (1% risk):
    active accounts : 4
    combined funded : $27,500
    combined balance: $28,025
    total paid out  : $7,965

Outputs (written to strategy_analysis/results/multiacct/):
    multiacct_equity.html    interactive 3-panel Plotly chart
    index.html               same chart (convenience copy)
    multiacct_report.md      detailed account-by-account report
    multiacct_summary.json   machine-readable summary + payout ledger
    combined_equity.csv      combined balance/level/account-count series
    account_<n>_equity.csv   per-account balance/level series

The only external import is strategy_analysis/backtest_engine.py (trade
execution). Data parsing, the core-3 setup list, the Classic funded-account
simulation, the multi-account orchestration and the chart are all here.
"""
import os
import re
import sys
import json
import glob
import shutil
import html as _html
from collections import Counter
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

from backtest_engine import execute_setup   # trade execution (kept engine)

RESULTS = os.path.join(HERE, "strategy_analysis", "results")
OUT = os.path.join(RESULTS, "multiacct")

# --------------------------------------------------------------------------- #
# Account / plan rules (Classic $5K + add-ons)
# --------------------------------------------------------------------------- #
STATIC_DD = 0.08      # 8% static max loss from the account level -> HARD breach
DAILY_DD = 0.04       # 4% daily drawdown -> soft breach (monitored)
TARGET = 0.10         # 10% profit target -> payout eligibility
MIN_DAYS = 6          # minimum trading days before the first payout
BEST_DAY_SHARE = 0.25 # 25% rule: best single day <= 25% of total profit
SPLIT = 0.90          # 90% profit split from day one (add-on)
LADDER = [5000, 7500, 10000, 25000, 60000, 150000]   # scaling every 3 payouts
REFUND_AT = 2         # fee refund after the account's 2nd payout
START = date(2026, 2, 1)   # account #1 bought 1 Feb
END = date(2026, 9, 9)     # no replay beyond today (real data ends 8 Sep)
RISK = 0.01           # 1% of the account's level per trade (never increased)
MAX_ACCOUNTS = 40     # safety cap on the account-opening cascade

# --------------------------------------------------------------------------- #
# Core-3 setups (the ONLY trades the plan takes) — same schema the engine uses
# --------------------------------------------------------------------------- #
CORE3_SETUPS = {
    "Monday": [
        {"pair": "USDCAD", "time_ist": "09:15 PM", "time_gmt": "15:45",
         "entry_mode": "15min", "rr": 3.0, "expected_wr": 61.5},
    ],
    "Tuesday": [],   # no trade
    "Wednesday": [
        {"pair": "GBPUSD", "time_ist": "08:30 PM", "time_gmt": "15:00",
         "entry_mode": "15min", "rr": 3.0, "expected_wr": 61.5},
    ],
    "Thursday": [
        {"pair": "EURUSD", "time_ist": "05:00 PM", "time_gmt": "11:30",
         "entry_mode": "5min", "rr": 2.0, "expected_wr": 66.7},
    ],
    "Friday": [],    # no trade
}
CORE_KEYS = ["Monday | USDCAD | 09:15 PM", "Wednesday | GBPUSD | 08:30 PM",
             "Thursday | EURUSD | 05:00 PM"]
PAIRS = ("EURUSD", "GBPUSD", "USDCAD")
EMPTY_COLS = ["datetime", "date", "time", "time_str", "day_of_week",
              "open", "high", "low", "close"]


# --------------------------------------------------------------------------- #
# 1) UTF-16 MT5 HTML export parser (same output schema as the backtest engine)
# --------------------------------------------------------------------------- #
def parse_mt5_csv_like(content):
    """Build the engine's OHLCV DataFrame from TSV lines (date time O H L C)."""
    data = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 6:
            continue
        try:
            dt = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y.%m.%d %H:%M")
        except ValueError:
            try:
                dt = datetime.strptime(f"{parts[0]} {parts[1]}",
                                       "%Y.%m.%d %H:%M:%S")
            except ValueError:
                continue
        data.append({
            "datetime": dt,
            "date": dt.date(),
            "time": dt.time(),
            "time_str": dt.strftime("%H:%M"),
            "day_of_week": dt.strftime("%A"),
            "open": float(parts[2]), "high": float(parts[3]),
            "low": float(parts[4]), "close": float(parts[5]),
        })
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.drop_duplicates(subset=["datetime"], keep="first")
    return df


def parse_mt5_htm(path):
    """Parse an MT5 HTML history export (UTF-16 with BOM, UTF-16 BE or
    UTF-8/ANSI fallback) into the DataFrame schema the engine expects.

    MT5's File > Save As Report saves HTML tables in UTF-16; each <tr> row is
        <date time>  <open>  <high>  <low>  <close>  <volume>
    """
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

    lines = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.I | re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.I | re.S)
        if len(cells) < 6:
            continue
        cells = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in cells]
        m = re.match(r"(\d{4}\.\d{2}\.\d{2})\s+(\d{2}:\d{2})", cells[0])
        if not m:
            continue  # header / non-data row
        lines.append("\t".join([m.group(1), m.group(2)] + cells[1:5]))
    return parse_mt5_csv_like("\n".join(lines))


def empty_frame():
    return pd.DataFrame({c: [] for c in EMPTY_COLS})


def discover_htm_files(data_dir):
    """Map pair -> {tf: path} from the .htm/.html MT5 exports in `data_dir`.

    When several exports exist for the same pair+timeframe (old re-uploads),
    the one with the LATEST _YYYYMMDDHHMM end-stamp in the filename wins —
    after the repo cleanup this resolves to exactly the 4 keeper files.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.htm")) +
                   glob.glob(os.path.join(data_dir, "*.html")))
    out = {}
    for f in files:
        base = os.path.basename(f).upper()
        pair = next((p for p in PAIRS if p in base), None)
        tf = "_M15" in base and "m15" or ("_M5" in base and "m5" or None)
        if not (pair and tf):
            continue
        m = re.search(r"_(\d{12})\.(?:HTM|HTML)$", base)
        key = m.group(1) if m else "00000000000000"
        cur = out.setdefault(pair, {}).get(tf)
        if cur is None or key > cur[0]:
            out.setdefault(pair, {})[tf] = (key, f)
    return {p: {t: v[1] for t, v in d.items()} for p, d in out.items()}


def load_pairs():
    """Load the 4 keeper .htm exports -> {pair: {'m5': df, 'm15': df}}.

    M15 is required for every pair (the ORB mother candle). M5 is only needed
    for the Thursday EURUSD 5-minute breakout; pairs without an M5 upload get
    an empty M5 frame (their setups never touch it).
    """
    files = discover_htm_files(HERE)
    pairs = {}
    for pair in PAIRS:
        if pair not in files or "m15" not in files[pair]:
            sys.exit(f"ERROR: missing M15 export for {pair} in {HERE} — "
                     f"cannot run the plan. Expected one of the keeper .htm files.")
        frames = {}
        for tf, f in sorted(files[pair].items()):
            df = parse_mt5_htm(f)
            spacing = median_spacing_min(df)
            print(f"  {pair:<7} {tf:<4} {os.path.basename(f):<45} rows={len(df):>7}"
                  f"  spacing={spacing}min  {df.datetime.min()} -> {df.datetime.max()}")
            frames[tf] = df
        if "m5" not in frames:
            frames["m5"] = empty_frame()
            print(f"  NOTE   {pair}: no M5 export — 15-min setups only, using empty M5 frame.")
        pairs[pair] = frames
    return pairs


def median_spacing_min(df):
    if df.empty or len(df) < 2:
        return None
    deltas = Counter()
    prev = None
    for dt in df["datetime"].sort_values():
        if prev is not None:
            m = (dt - prev).total_seconds() / 60.0
            if 0 < m < 200:
                deltas[m] += 1
        prev = dt
    return deltas.most_common(1)[0][0] if deltas else None


# --------------------------------------------------------------------------- #
# 2) Core-3 signal generation (engine executes each ORB setup per day)
# --------------------------------------------------------------------------- #
def generate_core3_signals(pairs_data, buffer_pips=2):
    """Run the 3 core setups over every trading date in the .htm window and
    return a DataFrame of trades (same columns as backtest_engine trades)."""
    all_dates = set()
    for p, d in pairs_data.items():
        if "m15" in d:
            all_dates.update(d["m15"]["date"].unique())

    trades = []
    for day in sorted(all_dates):
        dow = day.strftime("%A")
        for setup in CORE3_SETUPS.get(dow, []):
            pair = setup["pair"].upper().replace("/", "").replace(" ", "")
            if pair not in pairs_data:
                continue
            pd_ = pairs_data[pair]
            if "m5" not in pd_ or "m15" not in pd_:
                continue
            t = execute_setup(setup, pd_["m5"], pd_["m15"], day, buffer_pips)
            if t and t["result"] != "NO_TRADE":
                trades.append(t)
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def setup_key(row):
    return f"{row.day_of_week} | {row.pair} | {row.orb_time_ist}"


def r_of(row):
    return row.rr_ratio if row.result == "WIN" else -1.0


def get_core_signals(pairs_data):
    """Signals as (date, R) pairs, filtered to the core-3 setup keys."""
    df = generate_core3_signals(pairs_data)
    df = df.copy()
    df["setup"] = df.apply(setup_key, axis=1)
    df["R"] = df.apply(r_of, axis=1)
    sub = df[df.setup.isin(CORE_KEYS)].sort_values("entry_time").reset_index(drop=True)
    sub["day"] = pd.to_datetime(sub["entry_time"]).dt.date
    seq = list(zip(sub["day"], sub["R"]))
    print(f"\ncore-3 signals: {len(seq)} trades, {seq[0][0]} -> {seq[-1][0]}, "
          f"netR={sum(r for _, r in seq):.1f}")
    by_setup = sub.groupby("setup")["R"].agg(["count", "sum"])
    for k, g in by_setup.iterrows():
        print(f"    {k:<38} n={int(g['count']):>3}  netR={g['sum']:>6.1f}")
    return seq, sub


# --------------------------------------------------------------------------- #
# 3) Classic funded-account simulation (one account, one level)
# --------------------------------------------------------------------------- #
def simulate_account(seq, start, end, risk=RISK):
    """Simulate ONE Classic account trading `seq` from `start` to `end`.

    Every trade risks `risk` (1%) of the account's CURRENT LEVEL. Payouts
    reset the balance to the level; the 2nd payout refunds the fee (buys a new
    account) and every 3rd payout scales the account up the $5K ladder.
    """
    trades = [(d, R) for (d, R) in seq if start <= d <= end]
    events = []
    curve = []
    level_idx = 0
    level = LADDER[level_idx]
    balance = float(level)
    profit = 0.0
    best_day = 0.0          # best single day (25% payout rule)
    days = 0                # trading days since the last payout
    soft = 0                # 4% daily-drawdown breaches (soft)
    payout_no = 0
    breached = False
    for d, R in trades:
        risk_usd = risk * level
        pnl = risk_usd * R
        balance += pnl
        profit += pnl
        days += 1
        best_day = max(best_day, pnl)
        if pnl < -DAILY_DD * (balance - pnl):   # 4% of the pre-trade balance
            soft += 1
        curve.append((d, balance, level))
        if balance < level * (1 - STATIC_DD):   # 8% static max loss -> dead
            breached = True
            break
        if (profit >= TARGET * level             # 10% target hit,
                and best_day <= BEST_DAY_SHARE * profit   # 25% rule holds,
                and days >= MIN_DAYS):           # min 6 trading days
            payout_no += 1
            events.append({
                "payout_no": payout_no, "date": d, "level": level,
                "gross": round(profit, 2), "you": round(SPLIT * profit, 2),
                "split_pct": int(SPLIT * 100),
                "refund": payout_no == REFUND_AT,
                "scale": payout_no % 3 == 0,
                "new_level": (LADDER[min(payout_no // 3, len(LADDER) - 1)]
                              if payout_no % 3 == 0 else None),
            })
            level_idx = min(payout_no // 3, len(LADDER) - 1)
            level = LADDER[level_idx]
            balance = float(level)  # payout: balance resets to the level
            profit = 0.0
            best_day = 0.0
            days = 0
            soft = 0
    return {"events": events, "curve": curve, "breached": breached,
            "n_trades": len(trades)}


def balance_on(curve, D, default=float(LADDER[0])):
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


# --------------------------------------------------------------------------- #
# 4) Multi-account orchestration (reinvest fee refunds)
# --------------------------------------------------------------------------- #
def orchestrate(seq):
    accounts = []
    pending = [START]
    aid = 0
    while pending and aid < MAX_ACCOUNTS:
        sd = pending.pop(0)
        aid += 1
        res = simulate_account(seq, sd, END)
        accounts.append({"id": aid, "start": sd, **res})
        for e in res["events"]:
            if e["refund"] and e["date"] <= END:
                pending.append(e["date"])     # refund -> buy a new account
    print(f"Accounts opened by {END}: {len(accounts)}\n")

    master_dates = sorted({START} | {END} |
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

    print("=== PER-ACCOUNT SUMMARY ===")
    for a in accounts:
        ev = a["events"]
        last = (a["curve"][-1] if a["curve"]
                else (a["start"], float(LADDER[0]), float(LADDER[0])))
        paid = sum(e["you"] for e in ev)
        print(f"  Acct#{a['id']:<2} opened {a['start']}  trades={a['n_trades']:<3} "
              f"payouts={len(ev):<2} paidOut=${paid:>8,.2f}  "
              f"refundAt={next((str(e['date']) for e in ev if e['refund']), '-'):<12} "
              f"now: lvl=${last[2]:,.0f} bal=${last[1]:,.2f}  "
              f"{'BREACHED' if a['breached'] else ''}")

    print("\n=== PAYOUT LEDGER (1 Feb -> 9 Sep) ===")
    for p in payouts:
        note = []
        if p["refund"]:
            note.append("FEE REFUND -> new account")
        if p["scale"]:
            note.append(f"SCALE ${p['new_level']:,}")
        print(f"  {p['date']}  Acct#{p['account']:<2} P#{p['payout_no']}  "
              f"gross=${p['gross']:>7,.2f} {p['split_pct']}% you=${p['you']:>7,.2f}  "
              f"cum=${p['cumulative']:>9,.2f}  {', '.join(note)}")

    today = combined[-1]
    total_paid = payouts[-1]["cumulative"] if payouts else 0.0
    print("\n=== VERIFY — AS OF 9 SEP 2026 (1% risk) ===")
    print(f"  active accounts : {today['accounts']}")
    print(f"  combined funded : ${today['level']:,.0f}")
    print(f"  combined balance: ${today['balance']:,.2f}")
    print(f"  total paid out  : ${total_paid:,.2f}")
    if (today["accounts"] == 4 and today["level"] == 27500
            and today["balance"] == 28025 and total_paid == 7965):
        print("  -> MATCHES the verified plan numbers (4 / $27,500 / $28,025 / $7,965)")
    return accounts, combined, payouts, opens, today, total_paid


# --------------------------------------------------------------------------- #
# 5) Interactive 3-panel Plotly chart (equity+floor / payouts / accounts)
# --------------------------------------------------------------------------- #
def build_chart(combined, accounts, payouts, opens, today, total_paid):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.07, row_heights=[0.48, 0.26, 0.26],
        subplot_titles=("Combined equity vs 8% static max-loss floor "
                        "(dotted = per-account equity)",
                        "Cumulative payouts (90% split — you receive)",
                        "Active accounts (every fee refund buys a new $5K account)"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}],
               [{"secondary_y": False}]])

    cd = [r["date"] for r in combined]
    cb = [r["balance"] for r in combined]
    clv = [r["level"] for r in combined]
    ca = [r["accounts"] for r in combined]
    floor = [lv * (1 - STATIC_DD) for lv in clv]

    # --- row 1: combined equity + max-loss floor + per-account equity ---
    fig.add_trace(go.Scatter(
        x=cd, y=cb, name="Combined equity", mode="lines+markers",
        line=dict(color="#1a6ee0", width=2.5),
        hovertemplate=("Date: %{x|%d %b %Y}<br><b>Combined balance: $%{y:,.0f}</b><br>"
                       "Active accounts: %{customdata[0]}<br>"
                       "Combined funded: $%{customdata[1]:,.0f}<extra></extra>"),
        customdata=list(zip(ca, clv))), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cd, y=floor, name="8% max-loss floor (combined levels)",
        mode="lines", line=dict(color="#d62728", width=1.6, dash="dash"),
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

    # --- row 2: cumulative payouts + payout event dots ---
    px = [p["date"] for p in payouts]
    py = [p["cumulative"] for p in payouts]
    fig.add_trace(go.Scatter(
        x=px, y=py, name="Cumulative paid out (you)", mode="lines+markers",
        line=dict(color="#2ca02c", width=2), fill="tozeroy",
        hovertemplate="Date: %{x|%d %b %Y}<br>Cumulative paid out: $%{y:,.0f}"
                      "<extra></extra>"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=px, y=py, name="Payout event", mode="markers",
        marker=dict(color="#2ca02c", size=9, symbol="circle"),
        hovertemplate=("Acct #%{customdata[0]} · Payout %{customdata[1]}<br>"
                       "Date: %{x|%d %b %Y}<br>Level: $%{customdata[2]:,.0f}<br>"
                       "Gross: $%{customdata[3]:,.0f} · Split %{customdata[4]}%<br>"
                       "<b>You receive: $%{customdata[5]:,.0f}</b><br>"
                       "%{customdata[6]}<extra></extra>"),
        customdata=[[p["account"], p["payout_no"], p["level"], p["gross"],
                     p["split_pct"], p["you"],
                     ("FEE REFUND → new $5K acct · " if p["refund"] else "") +
                     (f"SCALE → ${p['new_level']:,}" if p["scale"] else "")]
                    for p in payouts]), row=2, col=1)

    # --- row 3: active accounts step + account-open stars ---
    fig.add_trace(go.Scatter(
        x=cd, y=ca, name="Active accounts", mode="lines",
        line=dict(color="#9467bd", width=2, shape="hv"),
        hovertemplate="Active accounts: %{y}<extra></extra>"), row=3, col=1)

    opens_sorted = sorted(opens, key=lambda o: o["date"])
    for i, o in enumerate(opens_sorted):
        y_after = i + 1   # each open raises the active count by one
        fig.add_trace(go.Scatter(
            x=[o["date"]], y=[y_after], name=f"Account #{o['account']} opened",
            mode="markers", marker=dict(color="#ff7f0e", size=12, symbol="star"),
            showlegend=(i == 0),
            hovertemplate=(f"Account #{o['account']} opened<br>"
                           "Date: %{x|%d %b %Y}<br>"
                           "Funded with the refunded fee · level $5,000"
                           "<extra></extra>")), row=3, col=1)

    for r in (1, 2, 3):
        fig.add_vline(x=datetime(2026, 9, 9), line_width=1.2, line_dash="dot",
                      line_color="#555", row=r, col=1)
    fig.add_annotation(x=datetime(2026, 9, 9), yref="paper", y=1.0,
                       text="TODAY 9 Sep — no future replay", showarrow=False,
                       xanchor="right", font=dict(size=11, color="#555"),
                       row=1, col=1)

    fig.update_yaxes(title_text="Balance (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Paid out (USD)", row=2, col=1)
    fig.update_yaxes(title_text="Accounts", row=3, col=1, range=[0,
                      max(ca) + 1])
    fig.update_xaxes(title_text="Date (2026)", row=3, col=1)
    fig.update_layout(
        title=("FundYourFX Classic $5,000 + add-ons (90% split, 8% max loss) — "
               "MULTI-ACCOUNT growth<br>"
               "<sup>Core-3 ORB: Mon USDCAD RR3 · Wed GBPUSD RR3 · Thu EURUSD RR2 "
               "| 1% risk/trade | every fee refund buys a new account | "
               "1 Feb → 9 Sep 2026</sup>"),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom",
                                           y=1.02),
        margin=dict(t=110), template="plotly_white", height=950)
    html = os.path.join(OUT, "multiacct_equity.html")
    fig.write_html(html, include_plotlyjs=True)
    shutil.copyfile(html, os.path.join(OUT, "index.html"))  # convenience copy
    print("Chart:", html)


# --------------------------------------------------------------------------- #
# 6) Markdown report
# --------------------------------------------------------------------------- #
def build_report(accounts, payouts, opens, today, total_paid):
    L = ["# Multi-Account Growth Report — Classic $5K + add-ons (1 Feb → 9 Sep 2026)",
         "",
         "> **Plan:** start ONE Classic $5K account on 1 Feb and trade ONLY the core-3 "
         "setups (Mon USDCAD 15:45 GMT RR3 · Wed GBPUSD 15:00 GMT RR3 · Thu EURUSD "
         "11:30 GMT RR2, 5-min breakout) at **1% risk** per trade. Every time an "
         "account's fee is refunded (after its 2nd payout), do NOT withdraw it — use "
         "it to buy a NEW $5K account that trades the same signals from that day. "
         "Add-ons: **90% split from day one**, **8% max loss**.",
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
        last = (a["curve"][-1] if a["curve"]
                else (a["start"], float(LADDER[0]), float(LADDER[0])))
        L.append(f"| #{a['id']} | {a['start']} | {a['n_trades']} | {len(ev)} | "
                 f"${paid:,.2f} | {refund} | ${last[2]:,.0f} | ${last[1]:,.2f} |")
    L += ["",
          "## Payout ledger (chronological)",
          "",
          "| Date | Account | Payout # | Gross | Split | You receive | Cumulative | Note |",
          "|---|---|---|---|---|---|---|---|"]
    for p in payouts:
        note = []
        if p["refund"]:
            note.append("**FEE REFUND → new account**")
        if p["scale"]:
            note.append(f"**SCALE → ${p['new_level']:,}**")
        L.append(f"| {p['date']} | #{p['account']} | {p['payout_no']} | "
                 f"${p['gross']:,.2f} | {p['split_pct']}% | ${p['you']:,.2f} | "
                 f"${p['cumulative']:,.2f} | {', '.join(note)} |")
    L += ["",
          "## How the accounts multiply",
          ""]
    for o in opens:
        if o["account"] == 1:
            L.append("- **1 Feb** — buy Account #1 with your own money ($5,000 Classic).")
        else:
            L.append(f"- **{o['date']}** — a fee refund lands → buy Account "
                     f"#{o['account']} and start trading the same signals.")
    L += ["",
          "## Interactive chart",
          "",
          "- Open `multiacct_equity.html` (or `index.html`) — three linked panels: "
          "combined equity vs the 8% max-loss floor (with per-account dotted "
          "equity), cumulative payouts (hover a dot for gross / split / your $), "
          "and active accounts (hover the stars for the account openings). Click "
          "legend items to toggle traces; hover any point for full details.",
          "",
          "## Assumptions",
          "",
          "1. **No future replay** — everything stops at 9 Sep 2026 (your real trade "
          "log ends 7 Sep; the .htm data ends 8 Sep).",
          "2. Each refund buys exactly one new $5K account (fee refund = price of one "
          "account).",
          "3. A new account starts trading the signals on/after its buy date (same "
          "trades as the older accounts that day).",
          "4. Split modeled as **90% flat** on every payout.",
          "5. Scaling (every 3 payouts) applies per account: $5K → $7.5K → $10K → "
          "$25K → $60K → $150K.",
          "6. Risk per trade = 1% of that account's current level (a scaled "
          "account's dollar risk grows with its level — that's the growth plan, "
          "not you raising risk).",
          "7. Funded-account rules modeled: 10% target, 8% static max loss (hard "
          "breach), 4% daily drawdown (soft), min 6 trading days, 25% best-day "
          "rule, fee refund after the 2nd payout.",
          ""]
    path = os.path.join(OUT, "multiacct_report.md")
    open(path, "w").write("\n".join(L))
    print("Report:", path)


# --------------------------------------------------------------------------- #
# 7) Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading the 4 keeper MT5 .htm exports (UTF-16)...")
    pairs = load_pairs()

    seq, sub = get_core_signals(pairs)
    if not seq:
        sys.exit("No core-3 signals found — check the .htm data coverage.")

    accounts, combined, payouts, opens, today, total_paid = orchestrate(seq)

    # ---- save data ----
    json.dump({"accounts": [{"id": a["id"], "start": str(a["start"]),
                             "n_trades": a["n_trades"], "breached": a["breached"],
                             "events": a["events"]} for a in accounts],
               "payouts": payouts,
               "opens": [{"account": o["account"], "date": str(o["date"])}
                         for o in opens],
               "as_of_today": {"date": "2026-09-09", "accounts": today["accounts"],
                               "combined_funded": today["level"],
                               "combined_balance": today["balance"],
                               "total_paid_out": total_paid}},
              open(os.path.join(OUT, "multiacct_summary.json"), "w"),
              indent=2, default=str)
    pd.DataFrame(combined).to_csv(os.path.join(OUT, "combined_equity.csv"),
                                  index=False)
    for a in accounts:
        pd.DataFrame([{"date": str(d), "balance": b, "level": lv}
                      for d, b, lv in a["curve"]]
                     ).to_csv(os.path.join(OUT, f"account_{a['id']}_equity.csv"),
                              index=False)

    build_chart(combined, accounts, payouts, opens, today, total_paid)
    build_report(accounts, payouts, opens, today, total_paid)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
