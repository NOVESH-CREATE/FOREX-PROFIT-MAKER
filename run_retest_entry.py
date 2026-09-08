"""
ORB Scenario-3 with LIMIT-RETEST entries (vs the original close-breakout entries).

Entry rule (everything else identical to the original backtest engine):
  1. Mother (ORB) candle = the 15-min candle at the setup time; range = high-low.
  2. The first candle that CLOSES beyond the range decides direction
     (same "close beyond" break rule the original backtest uses):
        close > mother high  -> LONG bias
        close < mother low   -> SHORT bias
  3. We do NOT enter on that break candle. After it closes we place a limit order:
        LONG  -> BUY LIMIT  at the mother high
        SHORT -> SELL LIMIT at the mother low
  4. The limit only fills if price retraces to the level the same day
     (long: a later candle's low <= mother high; short: high >= mother low).
     If price never comes back, the order expires at EOD -> no trade.
  5. SL/TP exactly as the original: SL at the opposite side of the mother candle
     +/- 2-pip buffer, TP = risk x RR (setup's rr_ratio unchanged).
  6. On the fill candle SL is checked before TP (conservative, same intra-candle
     convention as the engine); afterwards the original simulate_trade() runs.

Run: python3 run_retest_entry.py

Compares four runs side by side:
  - In-sample  (Aug 2025 - Jan 2026, original uploaded CSVs)  : original entry vs retest entry
  - Forward    (Feb - Sep 2026, uploaded .htm)                : original entry vs retest entry
"""
import sys, os, json
from datetime import datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

from backtest_engine import (get_pip_value, simulate_trade,
                             calculate_backtest_statistics, SCENARIO_3_SETUPS,
                             backtest_scenario_3)
from run_forward_uploads import discover_files, load_frame, empty_frame
from run_backtest import load_pairs

RESULTS_DIR = os.path.join(HERE, "strategy_analysis", "results")


# --------------------------------------------------------------------------- #
# Retest entry engine
# --------------------------------------------------------------------------- #
def simulate_from_fill(candles, fill_candle, entry, sl, tp, direction, pip_value):
    """Resolve a trade whose limit order filled on `fill_candle` (intra-bar)."""
    if direction == "LONG":
        if fill_candle["low"] <= sl:
            return {"result": "LOSS", "pnl_pips": round((sl - entry) / pip_value, 1),
                    "exit_time": fill_candle["datetime"], "exit_price": sl,
                    "exit_reason": "SL Hit"}
        if fill_candle["high"] >= tp:
            return {"result": "WIN", "pnl_pips": round((tp - entry) / pip_value, 1),
                    "exit_time": fill_candle["datetime"], "exit_price": tp,
                    "exit_reason": "TP Hit"}
    else:
        if fill_candle["high"] >= sl:
            return {"result": "LOSS", "pnl_pips": round((entry - sl) / pip_value, 1),
                    "exit_time": fill_candle["datetime"], "exit_price": sl,
                    "exit_reason": "SL Hit"}
        if fill_candle["low"] <= tp:
            return {"result": "WIN", "pnl_pips": round((entry - tp) / pip_value, 1),
                    "exit_time": fill_candle["datetime"], "exit_price": tp,
                    "exit_reason": "TP Hit"}

    # Candles strictly after the fill candle -> identical to the original engine.
    after = candles[candles["datetime"] > fill_candle["datetime"]]
    res = simulate_trade(after, entry, sl, tp, direction, pip_value)
    if res["result"] == "NO_TRADE":
        # Limit filled on the day's last candle -> close at that candle's close.
        px = fill_candle["close"]
        pnl = (px - entry) / pip_value if direction == "LONG" else (entry - px) / pip_value
        return {"result": "WIN" if pnl > 0 else "LOSS", "pnl_pips": round(pnl, 1),
                "exit_time": fill_candle["datetime"], "exit_price": px,
                "exit_reason": "EOD Close"}
    return res


def execute_setup_retest(setup, df_5min, df_15min, date, buffer_pips=2):
    pair = setup["pair"]
    orb_time_gmt = setup["time_gmt"]
    entry_mode = setup["entry_mode"]
    rr_ratio = setup["rr"]
    pip_value = get_pip_value(pair)
    buffer = buffer_pips * pip_value

    orb_candle = df_15min[(df_15min["date"] == date) &
                          (df_15min["time_str"] == orb_time_gmt)]
    if orb_candle.empty:
        return None
    orb = orb_candle.iloc[0]
    orb_high = orb["high"]
    orb_low = orb["low"]
    orb_datetime = orb["datetime"]
    orb_range_pips = (orb_high - orb_low) / pip_value
    if orb_range_pips < 2 or orb_range_pips > 100:
        return None

    if entry_mode == "5min":
        orb_end_time = orb_datetime + timedelta(minutes=15)
        day_candles = df_5min[(df_5min["date"] == date) &
                              (df_5min["datetime"] >= orb_end_time)].sort_values("datetime")
        breakout_df = df_5min
    else:
        day_candles = df_15min[(df_15min["date"] == date) &
                               (df_15min["datetime"] > orb_datetime)].sort_values("datetime")
        breakout_df = df_15min

    # 1) first CLOSE beyond the range sets direction
    direction = None
    break_candle = None
    for _, candle in day_candles.iterrows():
        if candle["close"] > orb_high:
            direction, break_candle = "LONG", candle
            break
        if candle["close"] < orb_low:
            direction, break_candle = "SHORT", candle
            break
    if direction is None:
        return None

    # 2) limit order + SL/TP (SL/TP math identical to the original)
    if direction == "LONG":
        entry_price = orb_high
        sl_price = orb_low - buffer
        risk = entry_price - sl_price
    else:
        entry_price = orb_low
        sl_price = orb_high + buffer
        risk = sl_price - entry_price
    if risk <= 0:
        return None
    tp_price = entry_price + risk * rr_ratio if direction == "LONG" else entry_price - risk * rr_ratio

    # 3) limit is live after the break candle closes; wait for the retrace
    after_break = breakout_df[(breakout_df["date"] == date) &
                              (breakout_df["datetime"] > break_candle["datetime"])].sort_values("datetime")
    fill_candle = None
    for _, candle in after_break.iterrows():
        if direction == "LONG":
            if candle["low"] <= entry_price:
                fill_candle = candle
                break
        else:
            if candle["high"] >= entry_price:
                fill_candle = candle
                break
    if fill_candle is None:
        return None  # never retraced to the level -> order expires, no trade

    result = simulate_from_fill(after_break, fill_candle, entry_price,
                                sl_price, tp_price, direction, pip_value)
    if result["result"] == "NO_TRADE":
        return None

    return {
        "date": date, "day_of_week": orb["day_of_week"], "pair": pair,
        "orb_time_gmt": orb_time_gmt, "orb_time_ist": setup["time_ist"],
        "entry_mode": entry_mode, "rr_ratio": rr_ratio,
        "expected_wr": setup.get("expected_wr", 0),
        "direction": direction,
        "orb_high": round(orb_high, 5), "orb_low": round(orb_low, 5),
        "orb_range_pips": round(orb_range_pips, 1),
        "break_time": break_candle["datetime"],
        "entry_time": fill_candle["datetime"],
        "entry_price": round(entry_price, 5),
        "sl_price": round(sl_price, 5),
        "tp_price": round(tp_price, 5),
        "risk_pips": round(risk / pip_value, 1),
        "result": result["result"], "pnl_pips": result["pnl_pips"],
        "exit_time": result["exit_time"], "exit_price": result["exit_price"],
        "exit_reason": result["exit_reason"],
    }


def backtest_scenario_3_retest(pairs_data, buffer_pips=2):
    all_trades = []
    all_dates = set()
    for pair, data in pairs_data.items():
        if "m15" in data:
            all_dates.update(data["m15"]["date"].unique())
    all_dates = sorted(all_dates)

    for date in all_dates:
        day_of_week = date.strftime("%A")
        if day_of_week in ("Saturday", "Sunday"):
            continue
        if day_of_week not in SCENARIO_3_SETUPS:
            continue
        for setup in SCENARIO_3_SETUPS[day_of_week]:
            pair = setup["pair"].upper().replace("/", "").replace(" ", "")
            if pair not in pairs_data:
                continue
            pair_data = pairs_data[pair]
            if "m5" not in pair_data or "m15" not in pair_data:
                continue
            trade = execute_setup_retest(setup, pair_data["m5"], pair_data["m15"],
                                         date, buffer_pips)
            if trade and trade["result"] != "NO_TRADE":
                all_trades.append(trade)
    return pd.DataFrame(all_trades)


# --------------------------------------------------------------------------- #
# Loading + reporting
# --------------------------------------------------------------------------- #
def load_forward_pairs():
    files = discover_files(HERE)
    pairs = {}
    for pair, d in sorted(files.items()):
        frames = {}
        for tf, f in sorted(d.items()):
            frames[tf] = load_frame(f)
        if "m15" not in frames:
            continue
        if "m5" not in frames:
            frames["m5"] = empty_frame()   # 15-min-only pairs never touch M5
        pairs[pair] = frames
    return pairs


def summarize_full(trades, label):
    if trades.empty:
        return None
    stats = calculate_backtest_statistics(trades)
    r = trades.apply(lambda row: row.rr_ratio if row.result == "WIN" else -1.0, axis=1)
    gp = float(r[r > 0].sum()); gl = float(abs(r[r < 0].sum()))
    s = {
        "label": label,
        "total_trades": stats["total_trades"], "wins": stats["wins"],
        "losses": stats["losses"], "win_rate": stats["win_rate"],
        "max_consecutive_wins": stats["max_consecutive_wins"],
        "max_consecutive_losses": stats["max_consecutive_losses"],
        "avg_rr": stats["avg_rr"], "net_R": round(float(r.sum()), 1),
        "gross_profit_R": round(gp, 1), "gross_loss_R": round(gl, 1),
        "profit_factor": round(gp / gl, 2) if gl else None,
        "total_pnl_pips": round(float(trades.pnl_pips.sum()), 1),
        "date_range": [str(trades.date.min()), str(trades.date.max())],
    }
    s["per_setup"] = {}
    t = trades.copy()
    t["setup"] = t.day_of_week + " | " + t.pair + " | " + t.orb_time_ist
    t["R"] = r.values
    for k, g in t.groupby("setup"):
        w = int((g.result == "WIN").sum())
        s["per_setup"][k] = {"trades": len(g), "wins": w, "losses": len(g) - w,
                             "win_rate": round(100 * w / len(g), 1),
                             "net_R": round(float(g.R.sum()), 1)}
    return s


def main():
    # ---- in-sample (original uploaded CSVs, as-is = "original conditions") ----
    pairs_is, _ = load_pairs()
    is_orig = backtest_scenario_3(pairs_is, buffer_pips=2)
    is_ret = backtest_scenario_3_retest(pairs_is, buffer_pips=2)

    # ---- forward (uploaded .htm, Feb -> Sep 2026) ----
    pairs_fwd = load_forward_pairs()
    fwd_orig = backtest_scenario_3(pairs_fwd, buffer_pips=2)
    fwd_ret = backtest_scenario_3_retest(pairs_fwd, buffer_pips=2)

    print(f"In-sample: original entry {len(is_orig)} trades | retest entry {len(is_ret)} trades")
    print(f"Forward  : original entry {len(fwd_orig)} trades | retest entry {len(fwd_ret)} trades")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(RESULTS_DIR, f"retest_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    is_orig.to_csv(os.path.join(outdir, "insample_original_trades.csv"), index=False)
    is_ret.to_csv(os.path.join(outdir, "insample_retest_trades.csv"), index=False)
    fwd_orig.to_csv(os.path.join(outdir, "forward_original_trades.csv"), index=False)
    fwd_ret.to_csv(os.path.join(outdir, "forward_retest_trades.csv"), index=False)

    s_is_orig = summarize_full(is_orig, "In-sample original entry")
    s_is_ret = summarize_full(is_ret, "In-sample retest entry")
    s_fwd_orig = summarize_full(fwd_orig, "Forward original entry")
    s_fwd_ret = summarize_full(fwd_ret, "Forward retest entry")

    json.dump({"insample_original": s_is_orig, "insample_retest": s_is_ret,
               "forward_original": s_fwd_orig, "forward_retest": s_fwd_ret},
              open(os.path.join(outdir, "retest_summary.json"), "w"), indent=2, default=str)

    metrics = [("total_trades", "Trades"), ("wins", "Wins"), ("losses", "Losses"),
               ("win_rate", "Win rate %"), ("net_R", "Net R"),
               ("gross_profit_R", "Gross profit R"), ("gross_loss_R", "Gross loss R"),
               ("profit_factor", "Profit factor"), ("avg_rr", "Avg RR"),
               ("total_pnl_pips", "Total pips"),
               ("max_consecutive_wins", "Max cons. wins"),
               ("max_consecutive_losses", "Max cons. losses")]

    def g(d, k):
        if d is None:
            return "-"
        v = d.get(k)
        return "-" if v is None else (round(v, 2) if isinstance(v, float) else v)

    md = ["# ORB Scenario-3: limit-retest entry vs original close-breakout entry",
          "",
          "## Entry rule change (everything else identical)",
          "1. Mother (ORB) candle = 15-min candle at setup time.",
          "2. First candle that **closes** beyond the range sets direction (same break rule as before).",
          "3. No entry on that candle. After it closes, place a limit order:",
          "   - close above mother high -> **BUY LIMIT at mother high**",
          "   - close below mother low  -> **SELL LIMIT at mother low**",
          "4. Order fills only if price retraces to the level same day; otherwise it expires at EOD (no trade).",
          "5. SL/TP unchanged: SL = opposite side of mother candle +/- 2 pips buffer, TP = risk x RR.",
          "6. On the fill candle SL is checked before TP; afterwards the original engine takes over.",
          "",
          "## Headline",
          "",
          "| Metric | In-sample: orig entry | In-sample: retest entry | Forward: orig entry | Forward: retest entry |",
          "|---|---|---|---|---|"]
    for k, label in metrics:
        md.append(f"| {label} | {g(s_is_orig, k)} | {g(s_is_ret, k)} | {g(s_fwd_orig, k)} | {g(s_fwd_ret, k)} |")

    md += ["", "## Per-setup — in-sample (Aug 2025 - Jan 2026)", "",
           "| Setup | Orig n | Orig WR% | Orig R | Retest n | Retest WR% | Retest R |",
           "|---|---|---|---|---|---|---|"]
    keys = list(s_is_orig["per_setup"].keys()) + [k for k in s_is_ret["per_setup"] if k not in s_is_orig["per_setup"]]
    for k in keys:
        a = s_is_orig["per_setup"].get(k, {}); b = s_is_ret["per_setup"].get(k, {})
        md.append(f"| {k} | {a.get('trades','-')} | {a.get('win_rate','-')} | {a.get('net_R','-')} | "
                  f"{b.get('trades','-')} | {b.get('win_rate','-')} | {b.get('net_R','-')} |")

    md += ["", "## Per-setup — forward (Feb - Sep 2026)", "",
           "| Setup | Orig n | Orig WR% | Orig R | Retest n | Retest WR% | Retest R |",
           "|---|---|---|---|---|---|---|"]
    keys = list(s_fwd_orig["per_setup"].keys()) + [k for k in s_fwd_ret["per_setup"] if k not in s_fwd_orig["per_setup"]]
    for k in keys:
        a = s_fwd_orig["per_setup"].get(k, {}); b = s_fwd_ret["per_setup"].get(k, {})
        md.append(f"| {k} | {a.get('trades','-')} | {a.get('win_rate','-')} | {a.get('net_R','-')} | "
                  f"{b.get('trades','-')} | {b.get('win_rate','-')} | {b.get('net_R','-')} |")

    open(os.path.join(outdir, "retest_report.md"), "w").write("\n".join(md))

    print("\n" + "\n".join(md))
    print("\nWrote:", os.path.join(outdir, "retest_report.md"))


if __name__ == "__main__":
    main()
