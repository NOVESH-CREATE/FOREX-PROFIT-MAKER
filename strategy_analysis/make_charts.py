"""Generate equity/monthly charts for the reproduced Scenario-3 backtest."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
trades = pd.read_csv(os.path.join(RES, "trades_scenario3.csv"))
trades["dt"] = pd.to_datetime(trades["entry_time"])
trades = trades.sort_values("dt").reset_index(drop=True)

# R-multiples (wins +RR, losses -1R)
r = np.where(trades.result == "WIN", trades.rr_ratio, -1.0)
cum_R = np.concatenate([[0.0], np.cumsum(r)])
dollar_fixed = 25000 + cum_R * 200.0   # $25k, $200 risk/trade (report config)
dates = pd.to_datetime(trades.date)

fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
ax.plot(dates, cum_R[1:], color="#0b7285", lw=1.8)
ax.fill_between(dates, cum_R[1:], 0, where=cum_R[1:] >= 0, color="#38d9a9", alpha=.3)
ax.fill_between(dates, cum_R[1:], 0, where=cum_R[1:] < 0, color="#ff8787", alpha=.3)
ax.axhline(0, color="grey", lw=.8)
ax.set_title("Scenario-3 cumulative profit (R-multiples) — Aug 2025 to Jan 2026", fontsize=12)
ax.set_ylabel("Net R (1 R = 1 SL distance)")
ax.grid(alpha=.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
fig.tight_layout()
fig.savefig(os.path.join(RES, "equity_R.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
ax.plot(dates, dollar_fixed[1:], color="#2b8a3e", lw=2)
ax.axhline(25000, color="grey", ls="--", lw=1)
ax.annotate("start $25,000", (dates.iloc[0], 25200), fontsize=9, color="grey")
ax.set_title("Equity curve, fixed $200 risk per trade (as in Feb-8 report)", fontsize=12)
ax.set_ylabel("Balance ($)")
ax.grid(alpha=.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
fig.tight_layout()
fig.savefig(os.path.join(RES, "equity_usd.png"))
plt.close(fig)

# Monthly bars in R
m = pd.to_datetime(trades.date).dt.to_period("M")
monthly = pd.DataFrame({"r": r, "m": m.astype(str)}).groupby("m").sum()["r"]
fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
colors = ["#2b8a3e" if v >= 0 else "#c92a2a" for v in monthly.values]
ax.bar(range(len(monthly)), monthly.values, color=colors)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly.index, rotation=45)
for i, v in enumerate(monthly.values):
    ax.text(i, v + (0.8 if v >= 0 else -2.2), f"{v:+.1f}R", ha="center", fontsize=9)
ax.axhline(0, color="grey", lw=.8)
ax.set_title("Net R by month", fontsize=12)
ax.set_ylabel("Net R")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(os.path.join(RES, "monthly_R.png"))
plt.close(fig)
print("charts saved")
