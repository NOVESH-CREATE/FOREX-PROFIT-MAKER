"""
data_audit.py — verify EVERY .htm data file before any backtest trusts it.
===========================================================================

Checks (run BEFORE any backtest so no mislabeled/buggy data can change results):

  1. Bar-spacing check  -> is the file genuinely M5 / M15? (the earlier
     in-sample EURUSD "M5" file was actually M15 — this check catches that)
  2. Coverage check     -> rows, first/last timestamp, duplicates, weekend bars
  3. Overlap check      -> old vs new EURUSD M5 exports must agree bar-for-bar
                           on OHLC wherever they overlap
  4. M5 -> M15 rebuild  -> aggregate the 5-min bars into 15-min bars and
                           compare to the REAL M15 export candle-by-candle.
                           This proves (a) the M5 file is really M5, (b) both
                           files share the same timezone, (c) the mother-candle
                           times line up — the exact failure mode that would
                           silently corrupt a 5-min-breakout backtest.
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from funded_plan import parse_mt5_htm          # the exact parser the plan uses
import pandas as pd


def spacing_profile(df):
    """Distribution of inter-bar gaps (minutes)."""
    dts = df["datetime"].sort_values()
    gaps = (dts.diff().dropna().dt.total_seconds() / 60).astype(int)
    gaps = gaps[(gaps > 0) & (gaps < 10080)]          # ignore weekend gaps
    return Counter(gaps)


def spacing_report(name, df):
    prof = spacing_profile(df)
    total = sum(prof.values())
    top = ", ".join(f"{g}m x{c} ({c/total*100:.1f}%)" for g, c in prof.most_common(4))
    return top


def file_report(tag, path):
    df = parse_mt5_htm(path)
    dups = len(df) - len(df.drop_duplicates(subset=["datetime"]))
    prof = spacing_profile(df)
    modal = prof.most_common(1)[0][0] if prof else None
    print(f"\n=== {tag} ===")
    print(f"  file      : {os.path.basename(path)}")
    print(f"  rows      : {len(df):,}  (duplicate timestamps: {dups})")
    print(f"  coverage  : {df.datetime.min()}  ->  {df.datetime.max()}")
    print(f"  gaps      : {spacing_report(tag, df)}")
    print(f"  modal gap : {modal} min  ->  "
          f"{'GENUINE M5' if modal == 5 else ('GENUINE M15' if modal == 15 else 'UNEXPECTED!')}")
    sat = df[df.datetime.dt.dayofweek >= 5]
    print(f"  weekend bars: {len(sat)}")
    return df, modal


def compare_frames(a, b, label, tol=1e-9):
    """Bar-for-bar OHLC comparison on the overlapping timestamps."""
    m = a.merge(b, on="datetime", suffixes=("_a", "_b"))
    if m.empty:
        print(f"  [{label}] NO OVERLAP")
        return
    bad = 0
    per_col = {}
    for c in ("open", "high", "low", "close"):
        d = (m[f"{c}_a"] - m[f"{c}_b"]).abs()
        n = int((d > tol).sum())
        per_col[c] = n
        bad += n
    verdict = "IDENTICAL" if bad == 0 else f"{bad} mismatched bars"
    print(f"  [{label}] overlap bars: {len(m):,}  ->  {verdict}")
    if bad:
        print(f"    mismatches per column: {per_col}")
        sample = m.loc[(m.open_a - m.open_b).abs() > tol
                       | (m.high_a - m.high_b).abs() > tol
                       | (m.low_a - m.low_b).abs() > tol
                       | (m.close_a - m.close_b).abs() > tol].head(3)
        print(sample.to_string(index=False))
    return len(m), bad


def rebuild_m15(df5):
    """Aggregate genuine 5-min bars into 15-min bars (label = bucket OPEN time)."""
    g = df5.set_index("datetime").resample("15min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last")).dropna()
    return g.reset_index()


def m5_vs_m15_check(pair, df5, df15):
    """Compare M5-rebuilt 15-min bars against the real M15 export."""
    r15 = rebuild_m15(df5)
    real = df15[["datetime", "open", "high", "low", "close"]].copy()
    m = real.merge(r15, on="datetime", suffixes=("_real", "_rebuilt"))
    if m.empty:
        print(f"  [{pair}] M5-vs-M15: NO OVERLAP  <-- TIMEZONE/LABEL BUG!")
        return None
    tol = 5e-5   # half a pip — some feeds round differently on rebuilds
    col_bad = {}
    for c in ("open", "high", "low", "close"):
        d = (m[f"{c}_real"] - m[f"{c}_rebuilt"]).abs()
        col_bad[c] = int((d > tol).sum())
    exact = int(((m.open_real == m.open_rebuilt) & (m.high_real == m.high_rebuilt) &
                 (m.low_real == m.low_rebuilt) & (m.close_real == m.close_rebuilt)).sum())
    ok = ((m.open_real - m.open_rebuilt).abs() <= tol) & \
         ((m.high_real - m.high_rebuilt).abs() <= tol) & \
         ((m.low_real - m.low_rebuilt).abs() <= tol) & \
         ((m.close_real - m.close_rebuilt).abs() <= tol)
    close = int(ok.sum())
    print(f"  [{pair}] real-M15 vs M5-rebuilt-M15: {len(m):,} candles compared | "
          f"exact={exact:,}  within-0.5pip={close:,}  "
          f"mismatch per col={col_bad}")
    # mother-candle sanity: the specific ORB times must exist in both
    for t in ("15:45", "11:30", "15:00"):
        n_real = (real.datetime.dt.strftime("%H:%M") == t).sum()
        n_reb = (r15.datetime.dt.strftime("%H:%M") == t).sum()
        if n_real and n_reb:
            print(f"    {t} candles: real-M15={n_real:,}  rebuilt={n_reb:,}")
    return len(m), col_bad


def main():
    files = {
        "USDCAD M5 NEW (backtest+forward)": os.path.join(
            HERE, "USDCAD_M5_202509011740_202609112055.htm"),
        "EURUSD M5 NEW (backtest+forward)": os.path.join(
            HERE, "EURUSD_M5_202509011740_202609112055.htm"),
        "EURUSD M5 OLD (forward only)": os.path.join(
            HERE, "EURUSD_M5_202602012205_202609080000.htm"),
        "EURUSD M15 (forward only)": os.path.join(
            HERE, "EURUSD_M15_202602012200_202609080000.htm"),
        "USDCAD M15 (forward only)": os.path.join(
            HERE, "USDCAD_M15_202602012200_202609081845.htm"),
        "GBPUSD M15 (forward only)": os.path.join(
            HERE, "GBPUSD_M15_202602012200_202609080000.htm"),
    }
    frames = {}
    for tag, path in files.items():
        df, modal = file_report(tag, path)
        frames[tag] = df

    print("\n\n########## CROSS-CHECKS ##########")
    eu5_new = frames["EURUSD M5 NEW (backtest+forward)"]
    eu5_old = frames["EURUSD M5 OLD (forward only)"]
    eu15 = frames["EURUSD M15 (forward only)"]
    uc5_new = frames["USDCAD M5 NEW (backtest+forward)"]
    uc15 = frames["USDCAD M15 (forward only)"]

    compare_frames(eu5_old, eu5_new, "EUR M5 OLD vs EUR M5 NEW")
    m5_vs_m15_check("EURUSD", eu5_new, eu15)
    m5_vs_m15_check("USDCAD", uc5_new, uc15)


if __name__ == "__main__":
    main()
