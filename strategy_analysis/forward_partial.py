"""
PARTIAL forward test of the ORB Scenario-3 strategy.

Runs the exact same engine (backtest_engine.py) on out-of-sample data that is
currently available on GitHub mirrors of the same broker-format MT5 exports:

  EURUSD: full M5+M15 through 2026-05-22 (merged big mirror + 5000-bar tail)
  GBPUSD: M15 tail 2026-03-11 -> 2026-05-22 (M5 tail present but setups need M15)
  USDCAD: NO public mirror exists -> those 3 setups cannot be forward-tested yet

Because the original repo files had mislabeled timeframes (EURUSD 'M5' was M15
data, GBPUSD 'M15' was M5 data), this partial forward test also rebuilds TRUE
timeframe in-sample baselines (Aug 2025 - Jan 2026) for the affected setups:
  - GBPUSD true M15  = resampled from the original genuine 5-min file
  - EURUSD true M5   = mirror full M5 file (2025-01-09 -> 2026-05-08)

Run: python3 strategy_analysis/forward_partial.py
"""
import os
import sys
import json
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import parse_mt5_csv, backtest_scenario_3, calculate_backtest_statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "forward_partial")
MIRROR = "/home/user/quant_src/data"  # cloned mirror repo
os.makedirs(RES, exist_ok=True)


def load_repo_csv(name):
    return parse_mt5_csv(open(os.path.join(REPO, name), encoding="utf-8-sig").read())


def load_mirror_csv(name):
    return parse_mt5_csv(open(os.path.join(MIRROR, name), encoding="utf-8-sig").read())


def resample_5_to_15(df5):
    """Resample a genuine 5-min frame into aligned 15-min candles (UTC floors)."""
    d = df5.copy()
    d["dt15"] = d["datetime"].apply(
        lambda t: t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0))
    g = d.groupby("dt15", sort=True)
    out = pd.DataFrame({
        "datetime": g["datetime"].first(),
        "date": g["date"].first(),
        "time": g["time"].first(),
        "time_str": g["time_str"].first(),
        "day_of_week": g["day_of_week"].first(),
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
    }).reset_index(drop=True)
    return out


def slice_frame(df, lo=None, hi=None):
    d = df.copy()
    if lo:
        d = d[d.datetime >= lo]
    if hi:
        d = d[d.datetime <= hi]
    return d.reset_index(drop=True)


def summarize(trades, label):
    if trades.empty:
        print(f"{label}: NO TRADES")
        return None
    stats = calculate_backtest_statistics(trades)
    r = trades.apply(lambda row: row.rr_ratio if row.result == "WIN" else -1.0, axis=1)
    s = {"label": label, "n": len(trades), "wins": stats["wins"], "losses": stats["losses"],
         "win_rate": stats["win_rate"], "net_R": round(float(r.sum()), 1),
         "total_pnl_pips": round(float(trades.pnl_pips.sum()), 1),
         "date_range": [str(trades.date.min()), str(trades.date.max())]}
    s["by_setup"] = {}
    t = trades.copy()
    t["setup"] = t.day_of_week + "|" + t.pair + "|" + t.orb_time_ist
    for k, g in t.groupby("setup"):
        w = int((g.result == "WIN").sum())
        s["by_setup"][k] = {"trades": len(g), "wins": w, "losses": len(g) - w,
                            "win_rate": round(100 * w / len(g), 1),
                            "net_R": round(float((g.apply(lambda x: x.rr_ratio if x.result == 'WIN' else -1.0, axis=1)).sum()), 1)}
    print(f"\n=== {label} ===")
    print(json.dumps(s, indent=1, default=str))
    return s


def main():
    # ---------- source frames ----------
    # Original repo files
    orig_eur_m15 = load_repo_csv("EURUSDm_M15_202508010000_202601302145.csv")   # TRUE M15
    orig_gbp_m5 = load_repo_csv("GBPUSDm_M5_202508010000_202601302155.csv")     # TRUE M5 (mislabeled in file name)
    # Mirror files
    mir_eur_m15_big = load_mirror_csv("EURUSDm_M15_202205060315_202605082045.csv")
    mir_eur_m15_tail = load_mirror_csv("EURUSDm_M15_20220101000000_20260523062135.csv")
    mir_eur_m5_big = load_mirror_csv("EURUSDm_M5_202501091005_202605082055.csv")
    mir_eur_m5_tail = load_mirror_csv("EURUSDm_M5_20220101000000_20260523062135.csv")
    mir_gbp_m15_tail = load_mirror_csv("GBPUSDm_M15_20220101000000_20260523062135.csv")
    mir_gbp_m5_tail = load_mirror_csv("GBPUSDm_M5_20220101000000_20260523062135.csv")

    # TRUE-timeframe frames -----------------------------------------------
    gbp_m5_true = orig_gbp_m5                                     # genuine 5-min, Aug25-Jan26
    gbp_m15_true_is = resample_5_to_15(orig_gbp_m5)               # true 15-min for in-sample

    eur_m15_is = orig_eur_m15                                     # true 15-min in-sample
    eur_m5_is = slice_frame(mir_eur_m5_big, datetime(2025, 8, 1), datetime(2026, 1, 30, 21, 55))

    # merged forward EUR frames (big + tail, prefer later export on overlap)
    eur_m15_fwd_raw = pd.concat([mir_eur_m15_big, mir_eur_m15_tail]).drop_duplicates(
        subset="datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    eur_m5_fwd_raw = pd.concat([mir_eur_m5_big, mir_eur_m5_tail]).drop_duplicates(
        subset="datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    eur_m15_fwd = slice_frame(eur_m15_fwd_raw, datetime(2026, 2, 1))
    eur_m5_fwd = slice_frame(eur_m5_fwd_raw, datetime(2026, 2, 1))
    gbp_m15_fwd = slice_frame(mir_gbp_m15_tail, datetime(2026, 2, 1))
    gbp_m5_fwd = slice_frame(mir_gbp_m5_tail, datetime(2026, 2, 1))

    print("frame sizes:",
          "gbp_m5_true", len(gbp_m5_true), "gbp_m15_is", len(gbp_m15_true_is),
          "eur_m15_is", len(eur_m15_is), "eur_m5_is", len(eur_m5_is),
          "eur_m15_fwd", len(eur_m15_fwd), "eur_m5_fwd", len(eur_m5_fwd),
          "gbp_m15_fwd", len(gbp_m15_fwd), "gbp_m5_fwd", len(gbp_m5_fwd))

    # ---------- IN-SAMPLE on true timeframes (only EUR/GBP setups) ----------
    is_data = {
        "EURUSD": {"m15": eur_m15_is, "m5": eur_m5_is},
        "GBPUSD": {"m15": gbp_m15_true_is, "m5": gbp_m5_true},
    }
    is_trades = backtest_scenario_3(is_data, buffer_pips=2)
    is_trades.to_csv(os.path.join(RES, "insample_trueTF_trades.csv"), index=False)
    is_sum = summarize(is_trades, "IN-SAMPLE true-timeframe (Aug'25-Jan'26, EUR+GBP setups only)")

    # ---------- FORWARD on true timeframes ----------
    fwd_data = {
        "EURUSD": {"m15": eur_m15_fwd, "m5": eur_m5_fwd},
        "GBPUSD": {"m15": gbp_m15_fwd, "m5": gbp_m5_fwd},
    }
    fwd_trades = backtest_scenario_3(fwd_data, buffer_pips=2)
    fwd_trades.to_csv(os.path.join(RES, "forward_trades.csv"), index=False)
    fwd_sum = summarize(fwd_trades, "FORWARD (2026-02-09..2026-05-22, EUR+GBP setups only)")

    # ---------- original (flawed-label) in-sample baselines for the same setups ----------
    orig_all = pd.read_csv(os.path.join(REPO, "strategy_analysis", "results", "trades_scenario3.csv"))
    keep = ["Wednesday|GBPUSD|08:30 PM", "Thursday|EURUSD|05:00 PM", "Thursday|GBPUSD|09:15 AM"]
    o = orig_all.copy()
    o["setup"] = o.day_of_week + "|" + o.pair + "|" + o.orb_time_ist
    orig_sub = o[o.setup.isin(keep)]
    orig_sum = summarize(orig_sub, "IN-SAMPLE original backtest log (flawed labels) - same 3 setups")

    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump({"insample_trueTF": is_sum, "forward": fwd_sum,
                   "insample_original_log": orig_sum}, fh, indent=2, default=str)

    # ---------- comparison md ----------
    def rowset(s):
        if not s:
            return {}
        return {k: v for k, v in s.get("by_setup", {}).items()}
    lines = ["# Partial forward test - ORB Scenario 3 (data-available setups)",
             "",
             f"- In-sample TRUE-timeframe ({is_sum['date_range']}): {is_sum['n']} trades, WR {is_sum['win_rate']}%, net {is_sum['net_R']}R",
             f"- FORWARD ({fwd_sum['date_range']}): {fwd_sum['n']} trades, WR {fwd_sum['win_rate']}%, net {fwd_sum['net_R']}R",
             "",
             "| Setup | IS-trueTF n | IS-trueTF WR | IS-trueTF R | FWD n | FWD WR | FWD R |",
             "|---|---|---|---|---|---|---|"]
    for k in sorted(rowset(is_sum)):
        a = rowset(is_sum).get(k, {}); b = rowset(fwd_sum).get(k, {})
        lines.append(f"| {k} | {a.get('trades','-')} | {a.get('win_rate','-')} | {a.get('net_R','-')} | "
                     f"{b.get('trades','-')} | {b.get('win_rate','-')} | {b.get('net_R','-')} |")
    lines += ["", "## Full trade log (forward)", ""]
    cols = ["date", "day_of_week", "pair", "orb_time_ist", "direction", "rr_ratio", "result", "pnl_pips", "exit_reason"]
    if not fwd_trades.empty:
        lines.append(fwd_trades[cols].to_markdown(index=False))
    open(os.path.join(RES, "comparison.md"), "w").write("\n".join(lines))
    print("\nSaved:", RES)


if __name__ == "__main__":
    main()
