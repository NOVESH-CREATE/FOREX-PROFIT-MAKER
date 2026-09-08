"""
Reproduce the Scenario-3 ORB backtest EXACTLY as app.py defines it,
using the CSV files stored in the repo.

Usage:
    python3 strategy_analysis/run_backtest.py

Outputs (written under strategy_analysis/results/):
    trades_scenario3.csv        - full trade-by-trade log
    stats_summary.json          - headline statistics
    compounding.json            - fixed / milestone / full-compound runs
"""
import sys, os, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import (parse_mt5_csv, backtest_scenario_3,
                             calculate_backtest_statistics, RealTradesCompounding)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)

PAIR_FILES = {
    "EURUSD": ("EURUSDm_M15_202508010000_202601302145.csv",
               "EURUSDm_M5_202508010000_202601302145.csv"),
    "GBPUSD": ("GBPUSDm_M15_202508010000_202601302155.csv",
               "GBPUSDm_M5_202508010000_202601302155.csv"),
    "USDCAD": ("USDCADm_M15_202508010000_202601302145.csv",
               "USDCADm_M5_202508010000_202601302155.csv"),
}

def load_pairs():
    pairs_data = {}
    info = {}
    for pair, (f15, f5) in PAIR_FILES.items():
        m15 = parse_mt5_csv(open(os.path.join(REPO, f15), encoding="utf-8-sig").read())
        m5  = parse_mt5_csv(open(os.path.join(REPO, f5),  encoding="utf-8-sig").read())
        pairs_data[pair] = {"m15": m15, "m5": m5}
        info[pair] = {"m15_rows": len(m15), "m5_rows": len(m5),
                      "m15_first": str(m15.datetime.min()), "m15_last": str(m15.datetime.max())}
    return pairs_data, info

def net_r_series(trades: pd.DataFrame) -> pd.Series:
    """R-multiple per trade: wins count +RR, losses count -1R (app risks 1R per trade)."""
    def r_of(row):
        if row.result == "WIN":
            return row.rr_ratio
        return -1.0
    return trades.apply(r_of, axis=1)

def main():
    pairs_data, info = load_pairs()
    print("Loaded:", json.dumps(info, indent=2))

    trades = backtest_scenario_3(pairs_data, buffer_pips=2)
    print(f"\nTrades found: {len(trades)}")
    if trades.empty:
        print("NO TRADES"); sys.exit(1)

    trades.to_csv(os.path.join(RES, "trades_scenario3.csv"), index=False)

    stats = calculate_backtest_statistics(trades)
    summary = {
        "total_trades": stats["total_trades"], "wins": stats["wins"], "losses": stats["losses"],
        "win_rate": stats["win_rate"], "max_consecutive_wins": stats["max_consecutive_wins"],
        "max_consecutive_losses": stats["max_consecutive_losses"], "avg_rr": stats["avg_rr"],
    }

    # R-curve sanity + pip totals
    trades_sorted = trades.sort_values("entry_time").reset_index(drop=True)
    r = net_r_series(trades_sorted)
    gross_profit_R = round(float(r[r > 0].sum()), 1)
    gross_loss_R   = round(float(abs(r[r < 0].sum())), 1)
    summary.update({
        "gross_profit_R": gross_profit_R, "gross_loss_R": gross_loss_R,
        "net_R": round(float(r.sum()), 1),
        "profit_factor_R": round(gross_profit_R / gross_loss_R, 2) if gross_loss_R else None,
        "total_pnl_pips": round(float(trades_sorted.pnl_pips.sum()), 1),
    })

    # Day-wise
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    d = stats["day_stats"]
    summary["day_stats"] = {k: {"trades": int(v["trades"]), "wins": int(v["wins"]),
                                "losses": int(v["losses"]), "win_rate": round(float(v["win_rate"]),1)}
                            for k, v in d.reindex([x for x in day_order if x in d.index]).iterrows()}
    # Pair-wise
    p = stats["pair_stats"]
    summary["pair_stats"] = {k: {"trades": int(v["trades"]), "wins": int(v["wins"]),
                                 "losses": int(v["losses"]), "win_rate": round(float(v["win_rate"]),1)}
                            for k, v in p.iterrows()}

    # Per-setup
    t2 = trades.copy(); t2["setup"] = t2["day_of_week"] + "|" + t2["pair"] + "|" + t2["orb_time_ist"]
    per_setup = []
    for key, g in t2.groupby("setup", sort=False):
        wins = int((g.result == "WIN").sum())
        per_setup.append({"setup": key, "trades": int(len(g)), "wins": wins,
                          "losses": int(len(g)-wins), "win_rate": round(100*wins/len(g), 1)})
    summary["per_setup"] = per_setup

    # Monthly
    t3 = trades.copy(); t3["month"] = pd.to_datetime(t3["date"]).dt.to_period("M").astype(str)
    monthly = []
    for key, g in t3.groupby("month", sort=True):
        wins = int((g.result == "WIN").sum())
        monthly.append({"month": key, "trades": int(len(g)), "wins": wins,
                        "losses": int(len(g)-wins),
                        "win_rate": round(100*wins/len(g), 1),
                        "net_R": round(float(net_r_series(g.reset_index(drop=True)).sum()), 2)})
    summary["monthly"] = monthly

    # Consecutive-loss worst stretch by R
    seq = list(zip(trades_sorted.entry_time, trades_sorted.pair, trades_sorted.result))
    worst = cur = 0
    for s in seq:
        cur = cur + 1 if s[2] == "LOSS" else 0
        worst = max(worst, cur)
    summary["max_consecutive_losses"] = worst

    with open(os.path.join(RES, "stats_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Compounding projections
    compounding = {}
    for (cap, risk, method) in [("report_fixed_25k", 25000.0, "fixed"),
                                ("default_milestone_50", 50.0, "milestone"),
                                ("full_compound_50", 50.0, "full_compound")]:
        if cap.startswith("report"):
            calc = RealTradesCompounding(25000.0, 0.8)   # $200 fixed risk/trade
        else:
            calc = RealTradesCompounding(50.0, 10.0)
        res = calc.apply_to_real_trades(trades, method)
        if res:
            compounding[cap] = {k: res[k] for k in
                ["method","initial_capital","final_balance","total_profit","total_return_pct",
                 "peak_balance","max_drawdown","max_drawdown_pct","total_trades","wins","losses",
                 "win_rate","profit_factor","gross_profit","gross_loss"]}
    with open(os.path.join(RES, "compounding.json"), "w") as f:
        json.dump(compounding, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print("\nCompounding:", json.dumps(compounding, indent=2, default=str))

if __name__ == "__main__":
    main()
