"""
FundYourFX Instant Funding (Pro) — funded-account simulator on the ORB strategy.

Models the account rules on the actual forward trades (Feb - Sep 2026, uploaded
.htm) and in-sample (Aug 2025 - Jan 2026, CSVs):

  - Static max drawdown  : 8% from initial balance   -> HARD breach (account dead)
  - Daily drawdown       : 5% of prior EOD balance   -> soft breach (avoid)
  - Profit target        : 8% (payout eligibility)
  - 25% rule             : best single-day profit <= 25% of total profit
  - (SL/HFT/stacking are satisfied by the strategy's bracket orders / low freq.)

Trades are re-simulated at RR 1:1, 1:2 and 1:3 (RR changes TP distance, so trade
outcomes genuinely change), for BOTH entry styles:
  - original : enter on the breakout candle's close
  - retest   : buy/sell limit at mother high/low after the break candle closes

Then, for each configuration, it answers:
  Q1  all setups vs each single setup
  Q2  retest vs no-retest
  Q3  RR 1:3 vs 1:2 vs 1:1
  Q4  what risk per trade passes the funded account without breaching

Run: python3 run_funded_sim.py
"""
import sys, os, json
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import backtest_engine as BE
from run_retest_entry import execute_setup_retest
from run_backtest import load_pairs
from run_forward_uploads import discover_files, load_frame, empty_frame

RESULTS_DIR = os.path.join(HERE, "strategy_analysis", "results")

STATIC_DD = 0.08
DAILY_DD = 0.05
TARGET = 0.08
RISK_GRID = [0.005, 0.01, 0.02]          # 0.5%, 1%, 2% risk per trade
BOOT = 2000


# --------------------------------------------------------------------------- #
# Trade generation at a given RR and entry style
# --------------------------------------------------------------------------- #
def gen_trades(pairs_data, rr, entry):
    trades = []
    all_dates = set()
    for p, d in pairs_data.items():
        if "m15" in d:
            all_dates.update(d["m15"]["date"].unique())
    for date in sorted(all_dates):
        dow = date.strftime("%A")
        if dow in ("Saturday", "Sunday") or dow not in BE.SCENARIO_3_SETUPS:
            continue
        for setup in BE.SCENARIO_3_SETUPS[dow]:
            s = dict(setup)
            s["rr"] = rr
            pair = s["pair"].upper().replace("/", "").replace(" ", "")
            if pair not in pairs_data:
                continue
            pd_ = pairs_data[pair]
            if "m5" not in pd_ or "m15" not in pd_:
                continue
            if entry == "retest":
                t = execute_setup_retest(s, pd_["m5"], pd_["m15"], date, 2)
            else:
                t = BE.execute_setup(s, pd_["m5"], pd_["m15"], date, 2)
            if t and t["result"] != "NO_TRADE":
                trades.append(t)
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def load_forward_pairs():
    files = discover_files(HERE)
    pairs = {}
    for pair, d in sorted(files.items()):
        frames = {}
        for tf, f in sorted(d.items()):
            frames[tf] = load_frame(f)
        if "m15" not in frames:
            continue
        if "m5" not in frames:
            frames["m5"] = empty_frame()
        pairs[pair] = frames
    return pairs


def setup_key(row):
    return f"{row.day_of_week} | {row.pair} | {row.orb_time_ist}"


def r_of(row):
    return row.rr_ratio if row.result == "WIN" else -1.0


# --------------------------------------------------------------------------- #
# R-metrics of a trade sequence
# --------------------------------------------------------------------------- #
def r_metrics(df):
    if df.empty:
        return None
    df = df.copy()
    df["R"] = df.apply(r_of, axis=1)
    df["day"] = pd.to_datetime(df["entry_time"]).dt.date
    cum = df["R"].cumsum()
    min_cum = float(cum.min())
    maxdd_R = max(0.0, -min_cum)                       # drawdown from initial, in R
    # daily net R
    daily = df.groupby("day")["R"].sum()
    worst_day_R = float(daily.min())
    best_day_R = float(daily.max())
    wins = int((df.result == "WIN").sum())
    n = len(df)
    gp = float(df.loc[df.R > 0, "R"].sum())
    gl = float(abs(df.loc[df.R < 0, "R"].sum()))
    # max consecutive losses (in R and count)
    cons = 0; maxcons = 0; cur = 0.0; worst_cons_R = 0.0
    for v in df["R"]:
        if v < 0:
            cons += 1; cur += abs(v)
            maxcons = max(maxcons, cons)
            worst_cons_R = max(worst_cons_R, cur)
        else:
            cons = 0; cur = 0.0
    m = {
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(100 * wins / n, 1),
        "net_R": round(float(df["R"].sum()), 1),
        "profit_factor": round(gp / gl, 2) if gl else None,
        "maxdd_R": round(maxdd_R, 1),
        "worst_day_R": round(worst_day_R, 1),
        "best_day_R": round(best_day_R, 1),
        "max_cons_losses": maxcons,
        "worst_cons_R": round(worst_cons_R, 1),
        "date_range": [str(df["day"].min()), str(df["day"].max())],
    }
    # 25% payout rule (best day profit vs total profit)
    total_R = float(df["R"].sum())
    m["best_day_share_%"] = round(100 * best_day_R / total_R, 1) if total_R > 0 else None
    m["_days"] = daily  # for funded sim
    m["_blocks"] = [g["R"].tolist() for _, g in df.groupby("day")]
    return m


# --------------------------------------------------------------------------- #
# Funded-account simulation
# --------------------------------------------------------------------------- #
def sim_blocks(blocks, risk, target=TARGET, static_dd=STATIC_DD, daily_dd=DAILY_DD):
    """blocks: list of lists of R (one list = one trading day). Returns a dict."""
    bal = 1.0; init = 1.0
    min_bal = 1.0
    daily_breaches = 0
    trades = 0; days = 0
    for block in blocks:
        if not block:
            continue
        days += 1
        day_start = bal
        day_pnl = 0.0
        for R in block:
            trades += 1
            pnl = bal * risk * R
            bal += pnl
            day_pnl += pnl
            min_bal = min(min_bal, bal)
            if bal <= init * (1 - static_dd):
                return {"outcome": "HARD_BREACH", "balance": bal, "trades": trades,
                        "days": days, "daily_breaches": daily_breaches,
                        "max_dd_pct": round(100 * (1 - min_bal), 2)}
            if bal >= init * (1 + target):
                return {"outcome": "TARGET", "balance": bal, "trades": trades,
                        "days": days, "daily_breaches": daily_breaches,
                        "max_dd_pct": round(100 * (1 - min_bal), 2)}
        if day_pnl < -daily_dd * day_start:
            daily_breaches += 1
    return {"outcome": "NO_TARGET", "balance": bal, "trades": trades,
            "days": days, "daily_breaches": daily_breaches,
            "max_dd_pct": round(100 * (1 - min_bal), 2)}


def bootstrap_pass(blocks, risk, n=BOOT):
    blocks = [b for b in blocks if b]  # drop empty days
    idx = np.arange(len(blocks))
    n_pass = 0; n_strict = 0
    for _ in range(n):
        samp = [blocks[i] for i in np.random.choice(idx, size=len(idx), replace=True)]
        r = sim_blocks(samp, risk)
        if r["outcome"] == "TARGET":
            n_pass += 1
            if r["daily_breaches"] == 0:
                n_strict += 1
    return round(100 * n_pass / n, 1), round(100 * n_strict / n, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    pairs_is, _ = load_pairs()
    pairs_fwd = load_forward_pairs()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(RESULTS_DIR, f"funded_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    # ---- cache all trade frames ----
    frames = {}   # (period, entry, rr) -> df
    for period, pairs in [("in", pairs_is), ("fwd", pairs_fwd)]:
        for entry in ["orig", "retest"]:
            for rr in [1.0, 2.0, 3.0]:
                frames[(period, entry, rr)] = gen_trades(pairs, rr, entry)

    # ---- Q1/Q2/Q3: edge table for "all setups" ----
    rows = []
    for period, plabel in [("in", "In-sample"), ("fwd", "Forward")]:
        for entry in ["orig", "retest"]:
            for rr in [1.0, 2.0, 3.0]:
                m = r_metrics(frames[(period, entry, rr)])
                if m:
                    rows.append((plabel, entry, rr, m))
    print("\n=== ALL-SETUPS EDGE TABLE ===")
    hdr = f"{'Period':<9} {'Entry':<7} {'RR':>4} {'n':>4} {'WR%':>6} {'netR':>7} {'PF':>5} {'maxDD_R':>7} {'worstDay_R':>10} {'consL':>5} {'bestDay%25':>10}"
    print(hdr)
    for plabel, entry, rr, m in rows:
        print(f"{plabel:<9} {entry:<7} {rr:>4.0f} {m['n']:>4} {m['win_rate']:>6} {m['net_R']:>7} "
              f"{m['profit_factor']:>5} {m['maxdd_R']:>7} {m['worst_day_R']:>10} {m['max_cons_losses']:>5} "
              f"{str(m['best_day_share_%']):>10}")

    # ---- funded sim on the ALL portfolio (forward) ----
    print("\n=== FUNDED SIM — ALL SETUPS, FORWARD (Feb-Sep 2026) ===")
    print(f"{'Entry':<7} {'RR':>4} {'risk%':>6} {'outcome':<12} {'maxDD%':>7} {'dailyBr':>7} {'trades':>6} {'days':>5} {'bootP(target)%':>14} {'bootP(strict)%':>14}")
    all_rows = []
    for entry in ["orig", "retest"]:
        for rr in [1.0, 2.0, 3.0]:
            m = r_metrics(frames[("fwd", entry, rr)])
            blocks = m["_blocks"]
            for risk in RISK_GRID:
                r = sim_blocks(blocks, risk)
                bp, bs = bootstrap_pass(blocks, risk)
                all_rows.append((entry, rr, risk, r, bp, bs))
                print(f"{entry:<7} {rr:>4.0f} {risk*100:>5.1f}% {r['outcome']:<12} {r['max_dd_pct']:>7} "
                      f"{r['daily_breaches']:>7} {r['trades']:>6} {r['days']:>5} {bp:>14} {bs:>14}")

    # ---- Q1: single setups vs all (forward, at RR 1:2 and 1:3) ----
    print("\n=== SINGLE-SETUP EDGE (forward) — orig vs retest, RR 1:2 and 1:3 ===")
    setup_rows = []
    for entry in ["orig", "retest"]:
        for rr in [2.0, 3.0]:
            df = frames[("fwd", entry, rr)]
            if df.empty:
                continue
            df = df.copy()
            df["setup"] = df.apply(setup_key, axis=1)
            df["R"] = df.apply(r_of, axis=1)
            for k, g in df.groupby("setup"):
                w = int((g.result == "WIN").sum())
                R = g["R"].sum()
                daily = g["R"].groupby(pd.to_datetime(g["entry_time"]).dt.date).sum()
                worst_day = float(daily.min())
                setup_rows.append({"entry": entry, "rr": rr, "setup": k, "n": len(g),
                                   "wins": w, "wr": round(100*w/len(g),1),
                                   "netR": round(float(R),1), "worst_day_R": round(worst_day,1)})
    for row in setup_rows:
        print(f"{row['entry']:<7} RR{row['rr']:.0f}  {row['setup']:<38} n={row['n']:>3} "
              f"WR={row['wr']:>5}%  netR={row['netR']:>7}  worstDay={row['worst_day_R']:>6}")

    # save everything
    json.dump({"all_edge": [{"period": p, "entry": e, "rr": r, **m} for p, e, r, m in rows],
               "funded_all": [{"entry": e, "rr": r, "risk": k, "outcome": o["outcome"],
                               "max_dd_pct": o["max_dd_pct"], "daily_breaches": o["daily_breaches"],
                               "trades": o["trades"], "days": o["days"], "boot_pass_pct": bp,
                               "boot_strict_pct": bs} for e, r, k, o, bp, bs in all_rows],
               "single_setups": setup_rows},
              open(os.path.join(outdir, "funded_summary.json"), "w"), indent=2, default=str)
    print("\nWrote:", os.path.join(outdir, "funded_summary.json"))


if __name__ == "__main__":
    main()
