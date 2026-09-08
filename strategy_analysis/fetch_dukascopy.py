#!/usr/bin/env python3
"""
Fetch clean, gap-free FX OHLCV from Dukascopy's free public datafeed and write
it as MetaTrader-5 style tab-separated CSV files (same format as the repo's
original exports), so the ORB forward-test harness can consume them directly.

Source (no API key, free): https://datafeed.dukascopy.com/datafeed/
  URL pattern (NOTE month is 0-indexed, January = 00):
    .../{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/BID_candles_min_1.bi5
  Format: LZMA-alone compressed, 24-byte big-endian records:
    struct ">IIIIIf" = seconds_from_day_start, open, close, low, high, volume
  Integer prices ÷ 100_000 for non-JPY 5-decimal pairs.

Output: per pair, both M5 and M15 resampled files, named to match the repo
convention so run_forward_test.py auto-discovers them, e.g.:
  forward_data/EURUSDm_M5_202602010000_202609081745.csv
  forward_data/EURUSDm_M15_202602010000_202609081745.csv  (etc.)

Resampling aligns on UTC boundaries (00:00, 00:05, ... 00:00, 00:15, ...) —
identical to MT5 bar times. Only bins that actually contain source minutes are
emitted, so there are no synthetic flat/gap candles.

Usage:
  python3 strategy_analysis/fetch_dukascopy.py --start 2026-02-01 --end 2026-09-08 \
      --out forward_data [--pairs EURUSD,GBPUSD,USDCAD]

Runs best inside GitHub Actions (full internet); can also run on any machine.
"""
import argparse
import lzma
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import date, datetime, timedelta

BASE = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">IIIIIf")  # offset_sec, open, close, low, high, volume
SCALE = 100_000.0                  # points per price unit for 5-decimal pairs
PAIRS = ["EURUSD", "GBPUSD", "USDCAD"]


def fetch_day(symbol: str, day: date, retries: int = 4) -> list:
    """Return list of (datetime_utc, open, high, low, close) 1-min candles or []."""
    url = f"{BASE}/{symbol}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    raw = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []  # no data published for that day
            if attempt == retries - 1:
                print(f"  ! {day} {symbol}: HTTP {e.code}", flush=True)
                return []
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! {day} {symbol}: {e}", flush=True)
                return []
            time.sleep(2 * (attempt + 1))
    if not raw:
        return []
    try:
        blob = lzma.decompress(raw)
    except Exception as e:
        print(f"  ! {day} {symbol}: lzma {e}", flush=True)
        return []
    day_start = datetime(day.year, day.month, day.day)
    rows = []
    for off, o, c, lo, hi, _v in RECORD.iter_unpack(blob):
        if o == 0:
            continue  # flat/empty placeholder
        ts = day_start + timedelta(seconds=off)
        rows.append((ts, o / SCALE, hi / SCALE, lo / SCALE, c / SCALE))
    return rows


def resample(rows, minutes):
    """Aggregate 1-min rows into OHLC on aligned boundaries of `minutes`."""
    bins = OrderedDict()
    for ts, o, h, l, c in rows:
        key = ts.replace(minute=(ts.minute // minutes) * minutes, second=0, microsecond=0)
        if key not in bins:
            bins[key] = [o, h, l, c, 1]
        else:
            b = bins[key]
            b[1] = max(b[1], h)
            b[2] = min(b[2], l)
            b[3] = c
            b[4] += 1
    return [(k, v[0], v[1], v[2], v[3], v[4]) for k, v in bins.items()]


def write_mt5_csv(path, bars):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\r\n")
        for ts, o, h, l, c, _v in bars:
            fh.write(f"{ts:%Y.%m.%d}\t{ts:%H:%M:%S}\t{o:.5f}\t{h:.5f}\t{l:.5f}\t{c:.5f}\t0\t0\t0\r\n")


def qa(path, minutes):
    """Print a QA line: row count, first/last, median spacing."""
    from collections import Counter
    deltas = Counter()
    prev = None
    rows = 0
    first = last = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("<") or not line.strip():
                continue
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 6:
                continue
            dt = datetime.strptime(f"{p[0]} {p[1]}", "%Y.%m.%d %H:%M:%S")
            rows += 1
            if first is None:
                first = dt
            last = dt
            if prev is not None:
                m = (dt - prev).total_seconds() / 60
                if m > 0 and m < 500:
                    deltas[m] += 1
            prev = dt
    spacing = deltas.most_common(1)[0][0] if deltas else None
    status = "OK" if spacing == minutes else "!! MISMATCH"
    print(f"  QA {os.path.basename(path)}: rows={rows} first={first} last={last} "
          f"spacing={spacing}min {status}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", default="forward_data")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    pairs = [p.upper() for p in args.pairs.split(",") if p.strip()]
    total_days = (end - start).days + 1
    print(f"Fetching {pairs} {start}..{end} ({total_days} calendar days) -> {args.out}", flush=True)

    all_rows = {p: [] for p in pairs}
    day = start
    while day <= end:
        if day.weekday() < 5:  # Mon-Fri only (Dukascopy has no Sat/Sun FX data)
            for p in pairs:
                rows = fetch_day(p, day)
                if rows:
                    all_rows[p].extend(rows)
                    print(f"  {day} {p}: {len(rows)} 1-min candles", flush=True)
                else:
                    print(f"  {day} {p}: no data", flush=True)
        day += timedelta(days=1)

    stamp = f"{start:%Y%m%d}000000_{end:%Y%m%d}235900"
    for p in pairs:
        if not all_rows[p]:
            print(f"!! no data at all for {p}")
            continue
        all_rows[p].sort(key=lambda r: r[0])
        for tf in ("M5", "M15"):
            bars = resample(all_rows[p], 5 if tf == "M5" else 15)
            path = os.path.join(args.out, f"{p}m_{tf}_{stamp}.csv")
            write_mt5_csv(path, bars)
            qa(path, 5 if tf == "M5" else 15)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
