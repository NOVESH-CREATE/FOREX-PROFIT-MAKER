"""Follow-up: native-RR configs + single-setup funded analysis (reuses run_funded_sim)."""
import sys, os, json
from datetime import datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import backtest_engine as BE
from run_retest_entry import execute_setup_retest
import run_funded_sim as FS

RESULTS_DIR = FS.RESULTS_DIR
RISK_GRID = FS.RISK_GRID


def gen_native(pairs_data, entry):
    """Generate trades using each setup's OWN rr (no override)."""
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
            pair = setup["pair"].upper().replace("/", "").replace(" ", "")
            if pair not in pairs_data:
                continue
            pd_ = pairs_data[pair]
            if "m5" not in pd_ or "m15" not in pd_:
                continue
            t = (execute_setup_retest(setup, pd_["m5"], pd_["m15"], date, 2) if entry == "retest"
                 else BE.execute_setup(setup, pd_["m5"], pd_["m15"], date, 2))
            if t and t["result"] != "NO_TRADE":
                trades.append(t)
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def single_metrics(df, setup):
    df = df.copy()
    df["setup"] = df.apply(FS.setup_key, axis=1)
    df["R"] = df.apply(FS.r_of, axis=1)
    g = df[df.setup == setup]
    if g.empty:
        return None
    sub = FS.r_metrics(g.reset_index(drop=True))
    return sub


def main():
    pairs_fwd = FS.load_forward_pairs()

    # native RR frames (forward)
    nat = {e: gen_native(pairs_fwd, e) for e in ["orig", "retest"]}

    print("=== NATIVE RR (forward, Feb-Sep 2026) ===")
    for e in ["orig", "retest"]:
        m = FS.r_metrics(nat[e])
        print(f"{e:<7} n={m['n']:>3} WR={m['win_rate']:>5}% netR={m['net_R']:>6} PF={m['profit_factor']} "
              f"maxDD_R={m['maxdd_R']} worstDay={m['worst_day_R']} consL={m['max_cons_losses']} "
              f"bestDay%={m['best_day_share_%']}%")

    # funded sim for native
    print("\n=== FUNDED SIM — NATIVE RR, FORWARD ===")
    print(f"{'Entry':<7} {'risk%':>6} {'outcome':<12} {'maxDD%':>7} {'dailyBr':>7} {'trades':>6} {'days':>5} {'bootP(target)%':>14} {'bootP(strict)%':>14}")
    for e in ["orig", "retest"]:
        m = FS.r_metrics(nat[e])
        for risk in RISK_GRID:
            r = FS.sim_blocks(m["_blocks"], risk)
            bp, bs = FS.bootstrap_pass(m["_blocks"], risk)
            print(f"{e:<7} {risk*100:>5.1f}% {r['outcome']:<12} {r['max_dd_pct']:>7} "
                  f"{r['daily_breaches']:>7} {r['trades']:>6} {r['days']:>5} {bp:>14} {bs:>14}")

    # single setups funded analysis (native RR, forward)
    singles = [
        "Wednesday | GBPUSD | 08:30 PM",   # RR 3.0
        "Thursday | EURUSD | 05:00 PM",    # RR 2.0
        "Monday | USDCAD | 09:15 PM",      # RR 3.0
        "Thursday | USDCAD | 04:45 PM",    # RR 2.0
        "Friday | USDCAD | 09:30 PM",      # RR 2.5
        "Thursday | GBPUSD | 09:15 AM",    # RR 2.0
    ]
    print("\n=== SINGLE-SETUP FUNDED (native RR, forward) ===")
    for e in ["orig", "retest"]:
        df = nat[e]
        for s in singles:
            m = single_metrics(df, s)
            if not m:
                print(f"{e:<7} {s:<38} (no trades)")
                continue
            # daily worst is -1R for a single setup (one trade/day); static binds via cons losses
            r1 = FS.sim_blocks(m["_blocks"], 0.01)
            bp1, bs1 = FS.bootstrap_pass(m["_blocks"], 0.01)
            print(f"{e:<7} {s:<38} n={m['n']:>3} WR={m['win_rate']:>5}% netR={m['net_R']:>6} "
                  f"maxDD_R={m['maxdd_R']:>4} consL={m['max_cons_losses']:>2} worstDay={m['worst_day_R']:>4} "
                  f"@1%: {r1['outcome']:<12} maxDD%={r1['max_dd_pct']:>5} dBr={r1['daily_breaches']} "
                  f"bootP={bp1:>5}% strict={bs1:>5}%")

    print("\ndone.")


if __name__ == "__main__":
    main()
