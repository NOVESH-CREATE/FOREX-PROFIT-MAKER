"""
Forward-test harness for the ORB Scenario-3 strategy, built for the *uploaded* .htm files.

Reads the MetaTrader-5 **HTML** export files that were uploaded to the repo
(EURUSD_M5 / EURUSD_M15 / GBPUSD_M15 / USDCAD_M15, forward period Feb 2026 onward),
parses them into the exact same DataFrame schema the backtest engine expects, then
runs the SAME engine (backtest_engine.py, extracted verbatim from app.py) on that
new, out-of-sample data and compares it against the original in-sample backtest.

Usage:
    python3 run_forward_uploads.py                 # scan the repo root for .htm/.csv uploads
    python3 run_forward_uploads.py --data <folder> # scan a specific folder

Notes on the uploaded set:
    - Every Scenario-3 setup needs the pair's M15 file (the ORB "mother" candle is
      always read from M15). All three pairs have an M15 upload here.
    - Only ONE setup uses 5-minute entries (Thursday EURUSD 05:00 PM IST). EURUSD is
      the only pair whose M5 file is required, and it is present.
    - GBPUSD and USDCAD trade only 15-minute setups, so their missing M5 uploads do
      not prevent a complete 6-setup forward test.

Outputs (strategy_analysis/results/forward_<timestamp>/):
    forward_trades.csv        trade-by-trade log of the forward period
    forward_stats.json        forward-period statistics + in-sample baseline
    comparison.md             side-by-side backtest vs forward table
"""
import sys, os, re, json, glob, argparse, html as _html
from collections import Counter
from datetime import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

from backtest_engine import (parse_mt5_csv, backtest_scenario_3,
                             calculate_backtest_statistics, SCENARIO_3_SETUPS)

RESULTS_DIR = os.path.join(HERE, "strategy_analysis", "results")
REPO_ROOT = HERE

EMPTY_COLS = ['datetime', 'date', 'time', 'time_str', 'day_of_week',
              'open', 'high', 'low', 'close']


# --------------------------------------------------------------------------- #
# HTML upload parser
# --------------------------------------------------------------------------- #
def parse_mt5_htm(path):
    """Parse an MT5 HTML export (UTF-16 or UTF-8) into the same DataFrame schema
    as backtest_engine.parse_mt5_csv. Reuses parse_mt5_csv for the field logic."""
    raw = open(path, 'rb').read()

    # MT5 HTML exports are usually UTF-16 (with BOM); fall back to UTF-8/ISO.
    if raw[:2] in (b'\xff\xfe',):
        text = raw.decode('utf-16-le', errors='replace')
    elif raw[:2] in (b'\xfe\xff',):
        text = raw.decode('utf-16-be', errors='replace')
    else:
        for enc in ('utf-8', 'cp1252', 'latin-1'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode('utf-8', errors='replace')

    lines = []
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', text, flags=re.I | re.S):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, flags=re.I | re.S)
        if len(cells) < 6:
            continue
        cells = [_html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in cells]
        m = re.match(r'(\d{4}\.\d{2}\.\d{2})\s+(\d{2}:\d{2})', cells[0])
        if not m:
            continue  # header row or non-data row
        # Emit the same tab-separated shape parse_mt5_csv already understands.
        lines.append("\t".join([m.group(1), m.group(2), cells[1], cells[2],
                                cells[3], cells[4]]))
    return parse_mt5_csv("\n".join(lines))


def parse_mt5_txt(path):
    return parse_mt5_csv(open(path, encoding='utf-8-sig', errors='replace').read())


def load_frame(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.htm', '.html'):
        return parse_mt5_htm(path)
    return parse_mt5_txt(path)


# --------------------------------------------------------------------------- #
# Discovery + QA
# --------------------------------------------------------------------------- #
def discover_files(data_dir):
    """Map pair -> {tf: path} for the uploaded forward data.

    HTML uploads (.htm/.html) take precedence: for any pair that has HTML
    uploads, ONLY the HTML uploads are used (the old in-sample .csv files that
    live in the repo root share pair/timeframe tokens and must not leak into the
    forward test). A pair is only read from .csv/.txt when it has no .htm/.html
    uploads at all (e.g. a fresh data folder with CSV exports)."""
    html_files = (sorted(glob.glob(os.path.join(data_dir, '*.htm'))) +
                  sorted(glob.glob(os.path.join(data_dir, '*.html'))))
    text_files = (sorted(glob.glob(os.path.join(data_dir, '*.csv'))) +
                  sorted(glob.glob(os.path.join(data_dir, '*.txt'))))

    def scan(files):
        # Keep, for each pair+timeframe, the export with the LATEST end
        # timestamp (parsed from the trailing _YYYYMMDDHHMM in the filename)
        # so re-uploads of the same pair/timeframe with extended coverage win.
        out = {}
        for f in files:
            base = os.path.basename(f).upper()
            pair = None
            for p in ('EURUSD', 'GBPUSD', 'USDCAD'):
                if p in base:
                    pair = p
                    break
            tf = None
            if '_M15' in base:
                tf = 'm15'
            elif '_M5' in base:
                tf = 'm5'
            if not (pair and tf):
                continue
            m = re.search(r'_(\d{12})\.(?:HTM|HTML|CSV|TXT)$', base)
            key = m.group(1) if m else '00000000000000'
            cur = out.setdefault(pair, {})
            if tf not in cur or key > cur[tf][0]:
                cur[tf] = (key, f)
        return {p: {t: v[1] for t, v in d.items()} for p, d in out.items()}

    htm = scan(html_files)
    txt = scan(text_files)
    out = {}
    for pair in set(list(htm.keys()) + list(txt.keys())):
        if pair in htm:
            out[pair] = htm[pair]          # HTML uploads win for this pair
        else:
            out[pair] = txt[pair]
    return out


def median_spacing_min(df):
    if df.empty or len(df) < 2:
        return None
    d = df['datetime'].sort_values()
    deltas = Counter()
    prev = None
    for dt in d:
        if prev is not None:
            m = (dt - prev).total_seconds() / 60.0
            if 0 < m < 200:
                deltas[m] += 1
        prev = dt
    return deltas.most_common(1)[0][0] if deltas else None


def empty_frame():
    return pd.DataFrame({c: [] for c in EMPTY_COLS})


# --------------------------------------------------------------------------- #
# Summary helpers (same shape as strategy_analysis/run_forward_test.py)
# --------------------------------------------------------------------------- #
def net_r_series(trades):
    return trades.apply(lambda row: row.rr_ratio if row.result == 'WIN' else -1.0, axis=1)


def summarize(trades):
    if trades.empty:
        return {'error': 'no trades'}
    stats = calculate_backtest_statistics(trades)
    r = net_r_series(trades)
    gp = float(r[r > 0].sum()); gl = float(abs(r[r < 0].sum()))
    s = {
        'total_trades': stats['total_trades'], 'wins': stats['wins'],
        'losses': stats['losses'], 'win_rate': stats['win_rate'],
        'max_consecutive_wins': stats['max_consecutive_wins'],
        'max_consecutive_losses': stats['max_consecutive_losses'],
        'avg_rr': stats['avg_rr'], 'net_R': round(float(r.sum()), 1),
        'gross_profit_R': round(gp, 1), 'gross_loss_R': round(gl, 1),
        'profit_factor': round(gp / gl, 2) if gl else None,
        'total_pnl_pips': round(float(trades.pnl_pips.sum()), 1),
        'date_range': [str(trades.date.min()), str(trades.date.max())],
    }
    s['per_setup'] = {}
    t = trades.copy()
    t['setup'] = t.day_of_week + ' | ' + t.pair + ' | ' + t.orb_time_ist
    for k, g in t.groupby('setup'):
        w = int((g.result == 'WIN').sum())
        s['per_setup'][k] = {'trades': len(g), 'wins': w, 'losses': len(g) - w,
                             'win_rate': round(100 * w / len(g), 1)}
    s['per_pair'] = {}
    for k, g in trades.groupby('pair'):
        w = int((g.result == 'WIN').sum())
        s['per_pair'][k] = {'trades': len(g), 'wins': w, 'losses': len(g) - w,
                            'win_rate': round(100 * w / len(g), 1)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=REPO_ROOT,
                    help='folder with the uploaded .htm/.csv MT5 exports (default: repo root)')
    args = ap.parse_args()

    files = discover_files(args.data)
    if not files:
        print('No .htm/.csv uploads found in', args.data)
        sys.exit(1)
    print('Discovered uploads:')
    for pair, d in sorted(files.items()):
        print('  ', pair, {k: os.path.basename(v) for k, v in sorted(d.items())})

    # Load + QA each upload.
    pairs_data = {}
    for pair, d in sorted(files.items()):
        frames = {}
        for tf, f in sorted(d.items()):
            df = load_frame(f)
            med = median_spacing_min(df)
            expect = 5 if tf == 'm5' else 15
            status = 'OK' if med == expect else '!! MISMATCH'
            span = f'{df.datetime.min()} -> {df.datetime.max()}' if not df.empty else 'EMPTY'
            print(f'  QA {pair} {tf}: {os.path.basename(f)}  rows={len(df)} '
                  f'spacing={med} min (expect {expect}) {status}  range={span}')
            frames[tf] = df
        pairs_data[pair] = frames

    # Every setup needs M15 (ORB mother candle). M5 is only needed for the
    # Thursday EURUSD 5-min breakout. Fill missing M5 for 15-min-only pairs so
    # the engine's key check passes, and hard-fail if a 5-min setup lacks M5.
    for pair, frames in pairs_data.items():
        needs_m5 = any(s['entry_mode'] == '5min'
                       for day in SCENARIO_3_SETUPS.values() for s in day
                       if s['pair'].upper().replace('/', '') == pair)
        if 'm15' not in frames:
            print(f'  ERROR {pair}: missing M15 upload - its setups cannot run.')
            sys.exit(1)
        if 'm5' not in frames:
            if needs_m5:
                print(f'  ERROR {pair}: has a 5-min setup but no M5 upload.')
                sys.exit(1)
            print(f'  NOTE {pair}: no M5 upload, but its setups are all 15-min '
                  f'(M5 unused) - proceeding with an empty M5 frame.')
            frames['m5'] = empty_frame()

    trades = backtest_scenario_3(pairs_data, buffer_pips=2)
    print(f'\nForward-period trades: {len(trades)}')
    if trades.empty:
        print('No trades - check date coverage / server clock of the uploads.')
        sys.exit(1)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(RESULTS_DIR, f'forward_{stamp}')
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir, 'forward_trades.csv'), index=False)

    fwd = summarize(trades)

    # Reload the in-sample baseline for side-by-side comparison.
    back = {}
    p = os.path.join(RESULTS_DIR, 'stats_summary.json')
    if os.path.exists(p):
        back = json.load(open(p))

    with open(os.path.join(outdir, 'forward_stats.json'), 'w') as fh:
        json.dump({'forward': fwd, 'backtest_period': back}, fh, indent=2, default=str)

    metrics = [('total_trades', 'Trades'), ('wins', 'Wins'), ('losses', 'Losses'),
               ('win_rate', 'Win rate %'), ('net_R', 'Net R'), ('avg_rr', 'Avg RR'),
               ('profit_factor', 'Profit factor'), ('total_pnl_pips', 'Total pips'),
               ('max_consecutive_wins', 'Max cons. wins'),
               ('max_consecutive_losses', 'Max cons. losses')]

    def get(d, k):
        if k == 'profit_factor':
            v = d.get('profit_factor', d.get('profit_factor_R'))
        else:
            v = d.get(k)
        return '-' if v is None else (round(v, 2) if isinstance(v, float) else v)

    def fmt_range(x):
        if isinstance(x, list) and len(x) == 2:
            return f'{x[0]} -> {x[1]}'
        return str(x) if x else ''

    rows = [f'| {label} | {get(back, k)} | {get(fwd, k)} |' for k, label in metrics]
    md = ['# ORB Scenario-3: forward test vs original backtest (uploaded .htm data)',
          '',
          f"- Original backtest (in-sample): {fmt_range(back.get('date_range')) or 'Aug 2025 - Jan 2026'}",
          f"- Forward test (out-of-sample uploads): {fmt_range(fwd.get('date_range'))}",
          '',
          '| Metric | Backtest (Aug 2025 - Jan 2026) | Forward (uploaded .htm) |',
          '|---|---|---|'] + rows + ['']

    def to_dict(x):
        # Normalise setup keys to a canonical "Day | Pair | HH:MM" form so the
        # in-sample list-of-dicts and the forward dict line up in the table.
        def canon(k):
            return re.sub(r'\s*\|\s*', ' | ', str(k).strip())
        if isinstance(x, dict):
            return {canon(k): v for k, v in x.items()}
        if isinstance(x, list):
            return {canon(i['setup']): {k: v for k, v in i.items() if k != 'setup'}
                    for i in x}
        return {}

    md.append('## Per-setup win rates')
    md.append('| Setup | BT trades | BT WR% | FWD trades | FWD WR% |')
    md.append('|---|---|---|---|---|')
    back_setup = to_dict(back.get('per_setup', {}))
    fwd_setup = to_dict(fwd.get('per_setup', {}))
    allkeys = list(back_setup.keys()) + [k for k in fwd_setup if k not in back_setup]
    for k in allkeys:
        b = back_setup.get(k, {})
        f = fwd_setup.get(k, {})
        md.append(f"| {k} | {b.get('trades','-')} | {b.get('win_rate','-')} | "
                  f"{f.get('trades','-')} | {f.get('win_rate','-')} |")

    open(os.path.join(outdir, 'comparison.md'), 'w').write('\n'.join(md))
    print('\nWrote:', os.path.join(outdir, 'comparison.md'))
    print(json.dumps(fwd, indent=2, default=str))


if __name__ == '__main__':
    main()
