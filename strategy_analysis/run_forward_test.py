"""
Forward-test harness for the ORB Scenario-3 strategy.

Runs the SAME engine as the original backtest (backtest_engine.py, extracted
verbatim from app.py) on NEW, out-of-sample data (Feb 2026 onward), then
compares the forward period against the backtest period.

Usage:
    python3 strategy_analysis/run_forward_test.py --data /path/to/folder

Expected input folder: six MetaTrader-5 CSV/TXT exports, same format as the
original repo files, e.g.:

    EURUSDm_M15_202602010000_202609081745.csv     EURUSDm_M5_202602010000_202609081745.csv
    GBPUSDm_M15_..._202609081755.csv              GBPUSDm_M5_...
    USDCADm_M15_...                               USDCADm_M5_...

Files are matched by the pair token (EURUSD / GBPUSD / USDCAD) and a _M5_ or
_M15_ token in the file name. The script will also CHECK that each M15 file is
actually 15-minute spaced and each M5 file is 5-minute spaced (the original
backtest files violated this for EURUSD and GBPUSD - it warns loudly here too,
so the forward test is comparable to what was actually backtested).

Outputs (strategy_analysis/results/forward_<timestamp>/):
    forward_trades.csv        trade-by-trade log of the forward period
    forward_stats.json        statistics for the forward period
    comparison.md             side-by-side backtest vs forward table
"""
import sys, os, re, json, glob, argparse
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import (parse_mt5_csv, backtest_scenario_3,
                             calculate_backtest_statistics, RealTradesCompounding)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def timeframe_check(path):
    """Return median candle spacing in minutes + note if data looks like another TF."""
    deltas = Counter()
    prev = None
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if line.startswith("<") or not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                dt = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y.%m.%d %H:%M:%S")
            except ValueError:
                continue
            if prev is not None:
                m = (dt - prev).total_seconds() / 60.0
                if m > 0 and m < 200:
                    deltas[m] += 1
            prev = dt
    if not deltas:
        return None
    med = deltas.most_common(1)[0][0]
    return med


def discover_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")) +
                   glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    out = {}
    for f in files:
        base = os.path.basename(f).upper()
        pair = None
        for p in ("EURUSD", "GBPUSD", "USDCAD"):
            if p in base:
                pair = p
                break
        tf = None
        if "_M15" in base:
            tf = "m15"
        elif "_M5" in base:
            tf = "m5"
        if pair and tf:
            out.setdefault(pair, {})[tf] = f
    return out


def net_r_series(trades):
    return trades.apply(lambda row: row.rr_ratio if row.result == "WIN" else -1.0, axis=1)


def summarize(trades):
    if trades.empty:
        return {"error": "no trades"}
    stats = calculate_backtest_statistics(trades)
    r = net_r_series(trades)
    gp = float(r[r > 0].sum()); gl = float(abs(r[r < 0].sum()))
    s = {
        "total_trades": stats["total_trades"], "wins": stats["wins"], "losses": stats["losses"],
        "win_rate": stats["win_rate"], "max_consecutive_wins": stats["max_consecutive_wins"],
        "max_consecutive_losses": stats["max_consecutive_losses"], "avg_rr": stats["avg_rr"],
        "net_R": round(float(r.sum()), 1), "gross_profit_R": round(gp, 1),
        "gross_loss_R": round(gl, 1),
        "profit_factor": round(gp / gl, 2) if gl else None,
        "total_pnl_pips": round(float(trades.pnl_pips.sum()), 1),
        "date_range": [str(trades.date.min()), str(trades.date.max())],
    }
    s["per_setup"] = {}
    t = trades.copy(); t["setup"] = t.day_of_week + " | " + t.pair + " | " + t.orb_time_ist
    for k, g in t.groupby("setup"):
        w = int((g.result == "WIN").sum())
        s["per_setup"][k] = {"trades": len(g), "wins": w, "losses": len(g) - w,
                             "win_rate": round(100 * w / len(g), 1)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the 6 new MT5 CSVs")
    args = ap.parse_args()

    files = discover_files(args.data)
    print("Discovered files:")
    for pair, d in files.items():
        print(" ", pair, {k: os.path.basename(v) for k, v in d.items()})

    # timeframe QA (same checks that the backtest files would have failed)
    for pair, d in files.items():
        for tf, f in d.items():
            med = timeframe_check(f)
            expect = 5 if tf == "m5" else 15
            status = "OK" if med == expect else "!! MISMATCH"
            print(f"  QA {pair} {tf}: median spacing = {med} min (expected {expect}) {status}")

    pairs_data = {}
    ok = True
    for pair, d in files.items():
        if "m5" not in d or "m15" not in d:
            print(f"  WARN {pair}: need both M5 and M15 - skipping")
            ok = False
            continue
        m15 = parse_mt5_csv(open(d["m15"], encoding="utf-8-sig", errors="replace").read())
        m5 = parse_mt5_csv(open(d["m5"], encoding="utf-8-sig", errors="replace").read())
        if m15.empty or m5.empty:
            print(f"  ERROR {pair}: files parsed to empty frames")
            ok = False
            continue
        pairs_data[pair] = {"m15": m15, "m5": m5}

    if not pairs_data:
        print("No usable data found."); sys.exit(1)

    trades = backtest_scenario_3(pairs_data, buffer_pips=2)
    print(f"\nForward-period trades: {len(trades)}")
    if trades.empty:
        print("No trades - check date coverage of the files.")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(RESULTS_DIR, f"forward_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir, "forward_trades.csv"), index=False)

    fwd = summarize(trades)
    # reload the in-sample summary for side-by-side
    back = {}
    p = os.path.join(RESULTS_DIR, "stats_summary.json")
    if os.path.exists(p):
        back = json.load(open(p))

    with open(os.path.join(outdir, "forward_stats.json"), "w") as fh:
        json.dump({"forward": fwd, "backtest_period": back}, fh, indent=2, default=str)

    rows = []
    metrics = [("total_trades", "Trades"), ("wins", "Wins"), ("losses", "Losses"),
               ("win_rate", "Win rate %"), ("net_R", "Net R"), ("avg_rr", "Avg RR"),
               ("profit_factor", "Profit factor"), ("total_pnl_pips", "Total pips"),
               ("max_consecutive_wins", "Max cons. wins"), ("max_consecutive_losses", "Max cons. losses")]
    def get(d, k):
        v = d.get(k)
        return "-" if v is None else (round(v, 2) if isinstance(v, float) else v)
    for k, label in metrics:
        rows.append(f"| {label} | {get(back, k)} | {get(fwd, k)} |")
    md = ["# ORB Scenario-3: forward test vs original backtest",
          "",
          f"- Original backtest (in-sample): {back.get('date_range')}",
          f"- Forward test (out-of-sample): {fwd.get('date_range')}",
          "",
          "| Metric | Backtest (Aug 2025 - Jan 2026) | Forward (new data) |",
          "|---|---|---|"] + rows + [""]
    # per-setup comparison
    def to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, list):
            return {i["setup"]: {k: v for k, v in i.items() if k != "setup"} for i in x}
        return {}

    md.append("## Per-setup win rates")
    md.append("| Setup | BT trades | BT WR% | FWD trades | FWD WR% |")
    md.append("|---|---|---|---|---|")
    back_setup = to_dict(back.get("per_setup", {}))
    fwd_setup = to_dict(fwd.get("per_setup", {}))
    allkeys = list(back_setup.keys()) + [k for k in fwd_setup if k not in back_setup]
    for k in allkeys:
        b = back_setup.get(k, {})
        f = fwd_setup.get(k, {})
        md.append(f"| {k} | {b.get('trades','-')} | {b.get('win_rate','-')} | {f.get('trades','-')} | {f.get('win_rate','-')} |")
    open(os.path.join(outdir, "comparison.md"), "w").write("\n".join(md))
    print("\nWrote:", os.path.join(outdir, "comparison.md"))
    print(json.dumps(fwd, indent=2, default=str))


if __name__ == "__main__":
    main()
