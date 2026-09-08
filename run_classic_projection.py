"""
FundYourFX Instant Funding (CLASSIC) $5,000 — payout / scaling / DD projection.

Uses the OFFICIAL Classic FAQ rules (NOT the Pro rules — they differ):

  - Profit target        : 10%
  - Max DD (static)      : 6%  (hard breach)
  - Daily DD             : 4% of EOD balance (soft)
  - Min trading days     : 6
  - Profit split         : 50% -> 60% -> 70% -> 80% -> 90% -> 95% (every 3 payouts)
  - Fee refund           : after 2 successful payouts
  - Growth plan          : every 3 payouts, 5K ladder:
        5,000 -> 7,500 -> 10,000 -> 25,000 -> 60,000 -> 150,000
  - 25% payout rule      : best single day <= 25% of total profit
  - Weekly payouts, $150 minimum

Strategy: core-3 setups (Mon USDCAD RR3, Wed GBPUSD RR3, Thu EURUSD RR2),
original close-breakout entry, 1 trade/day, risk % per trade.

Outputs: payout schedule (JSON/MD) + equity-curve chart (PNG) + curve CSV.
"""
import sys, os, json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import run_funded_sim as FS
import run_funded_native as FN

RESULTS_DIR = os.path.join(HERE, "strategy_analysis", "results")

# ---- Classic rules ----
STATIC_DD = 0.06
DAILY_DD = 0.04
TARGET = 0.10
MIN_DAYS = 6
LADDER_5K = [5000, 7500, 10000, 25000, 60000, 150000]
SPLITS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
CORE = ["Wednesday | GBPUSD | 08:30 PM", "Monday | USDCAD | 09:15 PM",
        "Thursday | EURUSD | 05:00 PM"]


def get_core_sequence():
    pairs = FS.load_forward_pairs()
    df = FN.gen_native(pairs, "orig").copy()
    df["setup"] = df.apply(FS.setup_key, axis=1)
    sub = df[df.setup.isin(CORE)].sort_values("entry_time").reset_index(drop=True)
    sub["R"] = sub.apply(FS.r_of, axis=1)
    sub["day"] = pd.to_datetime(sub["entry_time"]).dt.date
    return sub


def classic_pass_prob(seq_R, risks, boot=2000):
    """Bootstrap: P(hit 10% target before 6% static breach), zero daily breaches."""
    blocks = [[v] for v in seq_R]          # 1 trade/day
    idx = np.arange(len(blocks))
    out = {}
    for risk in risks:
        n_pass = n_strict = 0
        for _ in range(boot):
            samp = [blocks[i] for i in np.random.choice(idx, size=len(idx), replace=True)]
            r = FS.sim_blocks(samp, risk, target=TARGET, static_dd=STATIC_DD, daily_dd=DAILY_DD)
            if r["outcome"] == "TARGET":
                n_pass += 1
                if r["daily_breaches"] == 0:
                    n_strict += 1
        out[risk] = (round(100 * n_pass / boot, 1), round(100 * n_strict / boot, 1))
    return out


def next_date_after(d, weekday_iso):
    """Return the next date whose ISO weekday == weekday_iso (Mon=1..Fri=5)."""
    while d.isoweekday() != weekday_iso:
        d += timedelta(days=1)
    return d


def gen_dates(real_dates, n):
    """Infinite generator of trading dates: real ones first, then Mon/Wed/Thu cycle."""
    for d in real_dates:
        yield d
    last = real_dates[-1]
    iso = [1, 3, 4]   # Mon, Wed, Thu
    i = 0
    while True:
        last = next_date_after(last + timedelta(days=1), iso[i % 3])
        i += 1
        yield last


def project(seq_R, real_dates, risk_pct, max_cycles=18):
    """Deterministic multi-cycle projection. Returns rows + payout schedule + curve."""
    R_vals = list(seq_R)
    nR = len(R_vals)
    dates = gen_dates(real_dates, nR)
    curve = []          # (date, balance, level)
    schedule = []       # payout events
    cum_trader = 0.0
    ri = 0
    for c in range(max_cycles):
        lvl_idx = min(c // 3, len(LADDER_5K) - 1)
        level = LADDER_5K[lvl_idx]
        split = SPLITS[lvl_idx]
        balance = float(level)
        profit = 0.0
        best_day = 0.0
        days = 0
        floor = level * (1 - STATIC_DD)
        cycle_start_date = None
        soft_breaches = 0
        while True:
            R = R_vals[ri % nR]
            d = next(dates)
            ri += 1
            if cycle_start_date is None:
                cycle_start_date = d
            risk = balance * risk_pct
            pnl = risk * R
            prev_bal = balance
            balance += pnl
            profit += pnl
            days += 1
            best_day = max(best_day, pnl)
            # daily DD check (1 trade/day -> day pnl == pnl)
            if pnl < -DAILY_DD * prev_bal:
                soft_breaches += 1
            # static DD -> HARD breach
            if balance < floor:
                curve.append((d, balance, level, c, True))
                schedule.append({"payout": None, "date": str(d), "event": "HARD BREACH",
                                 "balance": round(balance, 2)})
                return curve, schedule, {"breached": True, "cycle": c + 1}
            curve.append((d, balance, level, c, False))
            # payout trigger
            if profit >= TARGET * level and best_day <= 0.25 * profit and days >= MIN_DAYS:
                trader = split * profit
                cum_trader += trader
                event = {
                    "payout": c + 1,
                    "date": str(d),
                    "level": level,
                    "gross_profit": round(profit, 2),
                    "profit_share_pct": int(split * 100),
                    "trader_payout": round(trader, 2),
                    "trader_cumulative": round(cum_trader, 2),
                    "trades_in_cycle": days,
                    "trading_days_ok": days >= MIN_DAYS,
                    "soft_breaches": soft_breaches,
                    "fee_refunded": (c + 1) == 2,
                    "scaling_after": (c + 1) % 3 == 0,
                    "new_level_after": (LADDER_5K[min((c + 1) // 3, len(LADDER_5K) - 1)]
                                        if (c + 1) % 3 == 0 else None),
                    "new_split_after": (int(SPLITS[min((c + 1) // 3, len(SPLITS) - 1)] * 100)
                                        if (c + 1) % 3 == 0 else None),
                }
                schedule.append(event)
                break
    return curve, schedule, {"breached": False}


def main():
    sub = get_core_sequence()
    seq_R = sub["R"].tolist()
    real_dates = list(sub["day"])
    print(f"Core-3 forward: {len(seq_R)} trades, netR={sum(seq_R):.1f}, "
          f"WR={100*(np.array(seq_R)>0).mean():.1f}%, "
          f"dates {real_dates[0]} -> {real_dates[-1]}")

    # 1) pass probability under Classic rules
    print("\n=== Classic bootstrap pass prob (core-3, target 10%, static DD 6%) ===")
    prob = classic_pass_prob(seq_R, [0.005, 0.01, 0.02])
    for r, (p, s) in prob.items():
        print(f"  risk {r*100:.1f}%  -> P(pass)={p}%  P(no daily breach)={s}%")

    # 2) deterministic projection at 1% and 0.5%
    for rp in [0.01, 0.005]:
        curve, schedule, meta = project(seq_R, real_dates, rp)
        print(f"\n=== PROJECTION at {rp*100:.1f}% risk/trade (breached={meta.get('breached')}) ===")
        for e in schedule:
            if e.get("event"):
                print(" ", e)
                continue
            print(f"  Payout#{e['payout']:<2} {e['date']}  level=${e['level']:>6}  "
                  f"gross=${e['gross_profit']:>7}  split={e['profit_share_pct']}%  "
                  f"you=${e['trader_payout']:>7}  cum=${e['trader_cumulative']:>8}  "
                  f"trades={e['trades_in_cycle']}  feeRefund={e['fee_refunded']}  "
                  f"scale={e['scaling_after']} newLvl={e['new_level_after']}")

        # save
        outdir = os.path.join(RESULTS_DIR, "classic_projection")
        os.makedirs(outdir, exist_ok=True)
        tag = f"{int(rp*1000)}"
        json.dump({"risk_pct": rp, "schedule": schedule, "breached": meta.get("breached")},
                  open(os.path.join(outdir, f"schedule_{tag}.json"), "w"), indent=2, default=str)

        # equity curve CSV
        rows = [{"date": str(d), "balance": round(b, 2), "level": lv, "cycle": c}
                for (d, b, lv, c, _) in curve]
        pd.DataFrame(rows).to_csv(os.path.join(outdir, f"equity_{tag}.csv"), index=False)

    # 3) chart (1% risk)
    curve, schedule, meta = project(seq_R, real_dates, 0.01)
    plot_projection(curve, schedule, 0.01)


def plot_projection(curve, schedule, risk_pct):
    dates = [d for (d, b, lv, c, _) in curve]
    bal = [b for (d, b, lv, c, _) in curve]
    lvl = [lv for (d, b, lv, c, _) in curve]
    dts = [datetime.combine(d, datetime.min.time()) for d in dates]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(dts, bal, lw=1.4, color="#1a6ee0", label="Account balance")
    # breach floor (per level)
    ax.step(dts, [lv * (1 - STATIC_DD) for lv in lvl], where="post",
            lw=1.2, ls="--", color="#d62728", alpha=0.7, label="6% static DD floor")

    # payout markers + scaling
    for e in schedule:
        if e.get("event"):
            continue
        d = datetime.combine(datetime.fromisoformat(e["date"]).date(), datetime.min.time())
        lv = e["level"]
        ax.scatter([d], [lv], color="#2ca02c", zorder=5, s=60)
        label = f"P{e['payout']} {e['profit_share_pct']}%"
        ax.annotate(label, (d, lv), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="#2ca02c")
        if e["scaling_after"]:
            ax.annotate(f"SCALE ${e['new_level_after']:,}", (d, lv),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=8, color="#9467bd", fontweight="bold")
        if e["fee_refunded"]:
            ax.annotate("FEE REFUND", (d, lv), textcoords="offset points",
                        xytext=(0, -30), ha="center", fontsize=8, color="#ff7f0e")

    ax.set_title(f"FundYourFX Classic $5,000 — core-3 ORB plan, {risk_pct*100:.0f}% risk/trade\n"
                 "(balance resets to level at each payout; scales every 3 payouts)")
    ax.set_ylabel("Balance (USD)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    outdir = os.path.join(RESULTS_DIR, "classic_projection")
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "equity_curve.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print("\nChart saved:", png)


if __name__ == "__main__":
    main()
