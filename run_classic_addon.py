"""
FundYourFX Classic $5,000 + ADD-ONS — date-accurate payout/scaling projection.

Add-ons (from user):
  - 90% profit split from day one
  - Max loss (static drawdown) raised to 8%

Classic rules (official FAQ, rest unchanged):
  - Profit target 10%, min 6 trading days, weekly payouts, min $150
  - Daily drawdown 4% of EOD balance (soft)
  - Fee refund after 2 successful payouts
  - Scaling every 3 payouts: 5,000 -> 7,500 -> 10,000 -> 25,000 -> 60,000 -> 150,000
  - 25% payout rule (best day <= 25% of total profit) -> effective target ~12%

Strategy: core-3 (Mon USDCAD RR3, Wed GBPUSD RR3, Thu EURUSD RR2), original
entry, 1 trade/day. Risk = fixed % of the account LEVEL per trade.

Run: python3 run_classic_addon.py
"""
import sys, os, json
from datetime import datetime, date, timedelta

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

# ---- rules (with add-ons) ----
STATIC_DD = 0.08          # add-on (was 6%)
DAILY_DD = 0.04
TARGET = 0.10
MIN_DAYS = 6
SPLIT = 0.90              # add-on (was 50% day one)
SPLIT_CAP = 0.95          # if the standard ladder still tops out at 95%
LADDER = [5000, 7500, 10000, 25000, 60000, 150000]
CORE = ["Wednesday | GBPUSD | 08:30 PM", "Monday | USDCAD | 09:15 PM",
        "Thursday | EURUSD | 05:00 PM"]


def get_core():
    pairs = FS.load_forward_pairs()
    df = FN.gen_native(pairs, "orig").copy()
    df["setup"] = df.apply(FS.setup_key, axis=1)
    sub = df[df.setup.isin(CORE)].sort_values("entry_time").reset_index(drop=True)
    sub["R"] = sub.apply(FS.r_of, axis=1)
    sub["day"] = pd.to_datetime(sub["entry_time"]).dt.date
    return sub


def _next_weekday(d, iso):
    while d.isoweekday() != iso:
        d += timedelta(days=1)
    return d


def date_stream(real_dates):
    """Yield real trading dates first, then continue Mon/Wed/Thu into the future."""
    yield from real_dates
    last = real_dates[-1]
    iso = [1, 3, 4]          # Monday, Wednesday, Thursday
    i = 0
    while True:
        last = _next_weekday(last + timedelta(days=1), iso[i % 3])
        i += 1
        yield last


def simulate(seq, risk_pct, split=SPLIT, split_cap=SPLIT_CAP, max_cycles=18,
             split_flat=True):
    """seq: list of (date, R). Fixed risk = risk_pct * level per trade.

    split_flat=True -> 90% for EVERY payout (literal add-on).
    split_flat=False -> 90% day one, 95% after 3 payouts (standard ladder cap)."""
    n = len(seq)
    real_dates = [d for d, _ in seq]
    dates = date_stream(real_dates)
    ri = 0
    events = []
    curve = []            # (date, balance, level, breached)
    cum_trader = 0.0
    for c in range(max_cycles):
        lvl_idx = min(c // 3, len(LADDER) - 1)
        level = LADDER[lvl_idx]
        if split_flat:
            s = split
        else:
            s = split if c < 3 else split_cap
        risk = risk_pct * level
        balance = float(level)
        profit = 0.0
        best_day = 0.0
        days = 0
        soft = 0
        for _ in range(n):
            R = seq[ri % n][1]
            d = next(dates)
            ri += 1
            pnl = risk * R
            balance += pnl
            profit += pnl
            days += 1
            best_day = max(best_day, pnl)
            if pnl < -DAILY_DD * (balance - pnl):
                soft += 1
            curve.append((d, balance, level, False))
            if balance < level * (1 - STATIC_DD):
                events.append({"payout": None, "date": str(d), "event": "HARD BREACH",
                               "balance": round(balance, 2), "level": level})
                return events, curve
            if profit >= TARGET * level and best_day <= 0.25 * profit and days >= MIN_DAYS:
                trader = s * profit
                cum_trader += trader
                events.append({
                    "payout": c + 1, "date": str(d), "level": level,
                    "gross_profit": round(profit, 2),
                    "split_pct": int(s * 100),
                    "trader_payout": round(trader, 2),
                    "trader_cumulative": round(cum_trader, 2),
                    "trades_in_cycle": days, "soft_breaches": soft,
                    "fee_refunded": (c + 1) == 2,
                    "scaling_after": (c + 1) % 3 == 0,
                    "new_level_after": (LADDER[min((c + 1) // 3, len(LADDER) - 1)]
                                        if (c + 1) % 3 == 0 else None),
                })
                break
    return events, curve


def bootstrap_pass(seq_R, risks, boot=2000):
    blocks = [[v] for v in seq_R]
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


def main():
    sub = get_core()
    seq = list(zip(sub["day"], sub["R"]))
    print(f"Core-3: {len(seq)} trades, netR={sum(r for _, r in seq):.1f}, "
          f"WR={100*np.mean([1 if r>0 else 0 for _, r in seq]):.1f}%")

    # pass probability with add-on DD
    print("\n=== Bootstrap pass prob (8% static DD add-on, 10% target) ===")
    for r, (p, s) in bootstrap_pass([R for _, R in seq], [0.005, 0.01, 0.02]).items():
        print(f"  risk {r*100:.1f}% -> P(pass)={p}%  P(no daily breach)={s}%")

    for rp in [0.01, 0.005]:
        events, curve = simulate(seq, rp)
        print(f"\n=== DATE-ACCURATE PROJECTION @ {rp*100:.1f}% risk/trade (fixed ${int(rp*5000)}) ===")
        print(f"{'#':<2} {'date':<12} {'level':>7} {'gross':>8} {'split':>5} {'you':>8} {'cum':>9}  note")
        for e in events:
            if e.get("event"):
                print(f"   {e['event']} on {e['date']} (level ${e['level']})")
                continue
            note = []
            if e["fee_refunded"]: note.append("FEE REFUND")
            if e["scaling_after"]: note.append(f"SCALE->${e['new_level_after']:,}")
            print(f"{e['payout']:<2} {e['date']:<12} {e['level']:>7} {e['gross_profit']:>8} "
                  f"{e['split_pct']:>4}% {e['trader_payout']:>8} {e['trader_cumulative']:>9}  "
                  f"{', '.join(note)}")

        # "as of today" (2026-09-09)
        cutoff = date(2026, 9, 9)
        done = [e for e in events if not e.get("event") and date.fromisoformat(e["date"]) <= cutoff]
        curve_today = [(d, b, lv) for (d, b, lv, br) in curve if d <= cutoff]
        cur_date, cur_bal, cur_lvl = curve_today[-1]
        total_paid = done[-1]["trader_cumulative"] if done else 0.0
        print(f"\n>>> AS OF TODAY (2026-09-09):")
        print(f"    payouts completed: {len(done)}   total paid out to you: ${total_paid:,.2f}")
        print(f"    current balance: ${cur_bal:,.2f}  (level ${cur_lvl:,.0f}) as of {cur_date}")

        # save
        outdir = os.path.join(RESULTS_DIR, "classic_addon")
        os.makedirs(outdir, exist_ok=True)
        tag = f"{int(rp*1000)}"
        json.dump({"risk_pct": rp, "events": events, "as_of_today": {
            "date": "2026-09-09", "payouts_completed": len(done),
            "total_paid_out": round(total_paid, 2),
            "current_balance": round(cur_bal, 2), "current_level": cur_lvl,
        }}, open(os.path.join(outdir, f"schedule_{tag}.json"), "w"), indent=2, default=str)
        pd.DataFrame([{"date": str(d), "balance": round(b, 2), "level": lv}
                      for (d, b, lv, br) in curve]).to_csv(os.path.join(outdir, f"equity_{tag}.csv"), index=False)

    plot(seq, 0.01)


def plot(seq, risk_pct):
    events, curve = simulate(seq, risk_pct)
    dates = [d for (d, b, lv, br) in curve]
    bal = [b for (d, b, lv, br) in curve]
    lvl = [lv for (d, b, lv, br) in curve]
    dts = [datetime.combine(d, datetime.min.time()) for d in dates]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.plot(dts, bal, lw=1.3, color="#1a6ee0", label="Account balance (compounds 1% risk)")
    ax.step(dts, [lv * (1 - STATIC_DD) for lv in lvl], where="post", lw=1.2, ls="--",
            color="#d62728", alpha=0.7, label="8% static DD floor (add-on)")

    for e in events:
        if e.get("event"):
            continue
        d = datetime.combine(date.fromisoformat(e["date"]), datetime.min.time())
        ax.scatter([d], [e["level"]], color="#2ca02c", zorder=5, s=55)
        ax.annotate(f"P{e['payout']} 90% ${e['trader_payout']:,.0f}", (d, e["level"]),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=7.5, color="#2ca02c")
        if e["scaling_after"]:
            ax.annotate(f"SCALE ${e['new_level_after']:,}", (d, e["level"]),
                        textcoords="offset points", xytext=(0, 15), ha="center",
                        fontsize=8, color="#9467bd", fontweight="bold")
        if e["fee_refunded"]:
            ax.annotate("FEE REFUND", (d, e["level"]), textcoords="offset points",
                        xytext=(0, -32), ha="center", fontsize=8, color="#ff7f0e")

    ax.set_title("FundYourFX Classic $5,000 + ADD-ONS (90% split, 8% max loss) — core-3 ORB plan\n"
                 f"1% risk/trade · payouts reset to level · scales every 3 payouts")
    ax.set_ylabel("Balance (USD)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    outdir = os.path.join(RESULTS_DIR, "classic_addon")
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "equity_curve.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print("\nChart saved:", png)


if __name__ == "__main__":
    main()
