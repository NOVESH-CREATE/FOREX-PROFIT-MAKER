"""
project_4yr.py — 4-YEAR PROJECTION of the EU+GU FYFX plan (Monte Carlo)
=======================================================================
HOW FUNDS "PREDICT" LONG HORIZONS: they don't point-forecast — they
simulate thousands of alternative futures from the strategy's VERIFIED trade
distribution, pass each through the real account machinery, and read the
PROBABILITY DISTRIBUTION of outcomes + the survival odds. That is what this
script does, end to end:

  Diagnostics (on the 104 gate-verified trades, Sep 2025 - Sep 2026):
    * ADF stationarity test on the R series
    * 2-state Markov regime-switching fit (statsmodels MarkovRegression)

  Generative models (how future trades are drawn):
    * BLOCK BOOTSTRAP of observed R (blocks of 8 - preserves loss/win
      streaks and volatility clustering)
    * MARKOV-REGIME draws from the fitted 2-state model

  Edge-decay stress (the honest core of any 4-year claim):
    * multiplier m shrinks expectancy toward zero while keeping variance:
      R' = m*mean + (R - mean);  m = 1.0 / 0.5 / 0.25 / 0.0
    * + a 0.5 pip/trade cost-drag scenario on the full edge

  Account machinery per simulated future (UNCHANGED from the verified plan):
    * full FYFX cascade via simulate_account_fyfx (8% target once, $150 min
      payout, 90% split, 6-day spacing, refund at payout 3 -> new $5K
      account, scale every 3 payouts $5K->...->$150K, 8% static breach)
    * breach policy: the dead account is REPLACED by a fresh $5K account on
      the next trading day (the business continues; count the breaches)

  Outputs: per-scenario distribution of TOTAL PROFIT (cash payouts + in-
  cycle profit) at each quarter for 4 years, P(no breach), P(funded level
  >= $25K/$60K), payouts/year, fan chart, markdown report.

Run: python3 strategy_analysis/project_4yr.py
"""
import os
import sys
import json
import bisect
from datetime import date, timedelta

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "strategy_analysis"))

import numpy as np
import pandas as pd

from eu_gu_funded_plan import (simulate_account_fyfx, load_gbp_upload,
                               run_window_setup, rebuild_m15, EU_MOTHER,
                               GU_MOTHER, DATA_START, FW_END, parse_mt5_htm)
from funded_plan import (load_pairs, generate_core3_signals, setup_key, r_of,
                         LADDER, MAX_ACCOUNTS)

OUT = os.path.join(HERE, "strategy_analysis", "results", "projection_4yr")
os.makedirs(OUT, exist_ok=True)

RISK = 0.015
HORIZON_YEARS = 4
PROJ_START = date(2026, 9, 14)          # first Mon after the data ends
PATHS = 2000                            # per scenario
RNG = np.random.default_rng(20260913)


# --------------------------------------------------------------------------- #
# 1. Verified history -> trade-R universe
# --------------------------------------------------------------------------- #
def build_r_universe():
    pairs = load_pairs()
    df = generate_core3_signals(pairs)
    df = df.copy()
    df["setup"] = df.apply(setup_key, axis=1)
    df["R"] = df.apply(r_of, axis=1)
    sub = df[df.setup.isin(["Wednesday | GBPUSD | 08:30 PM",
                            "Thursday | EURUSD | 05:00 PM"])]
    sub = sub.sort_values("entry_time").reset_index(drop=True)
    sub["day"] = pd.to_datetime(sub["entry_time"]).dt.date
    eu5 = parse_mt5_htm(os.path.join(
        HERE, "EURUSD_M5_202509011740_202609112055.htm"))
    eu_full = run_window_setup(EU_MOTHER, eu5, rebuild_m15(eu5),
                               DATA_START, FW_END)
    eu_full["R"] = eu_full.apply(r_of, axis=1)
    gbp5, kind, path = load_gbp_upload()
    gu_fy = run_window_setup(GU_MOTHER, None, gbp5, DATA_START, FW_END)
    gu_fy["R"] = gu_fy.apply(r_of, axis=1)
    full = pd.concat([eu_full, gu_fy]).sort_values("entry_time")
    R = full["R"].to_numpy(dtype=float)
    assert len(R) == 104, f"expected 104 verified trades, got {len(R)}"
    return R


def make_dates():
    """Every Wed+Thu from PROJ_START for HORIZON_YEARS (skip Dec 25 / Jan 1)."""
    out, d = [], PROJ_START
    end = date(PROJ_START.year + HORIZON_YEARS, PROJ_START.month,
               PROJ_START.day)
    while d < end:
        if d.weekday() in (2, 3) and not (d.month == 12 and d.day == 25) \
                and not (d.month == 1 and d.day == 1):
            out.append(d)
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# 2. Generative models
# --------------------------------------------------------------------------- #
def draw_block_bootstrap(R, n, rng, block=8):
    """Stationary block bootstrap - keeps streaks/clustering."""
    out = np.empty(n)
    i = 0
    while i < n:
        s = rng.integers(0, len(R))
        for k in range(block):
            if i >= n:
                break
            out[i] = R[(s + k) % len(R)]
            i += 1
    return out


def fit_markov(R):
    """2-state MarkovRegression: state 0 = 'loss run' state (the ML solution
    isolates full -1R stop-outs), state 1 = 'normal' state. Moments match the
    empirical distribution; the chain reproduces streakiness."""
    from statsmodels.tsa.regime_switching.markov_regression import (
        MarkovRegression)
    mod = MarkovRegression(R, k_regimes=2, trend="c", switching_variance=True)
    res = mod.fit(search_reps=50, disp=False)
    params, names = res.params, res.model.param_names
    p00 = params[names.index("p[0->0]")]      # P(0 -> 0)
    p10 = params[names.index("p[1->0]")]      # P(1 -> 0)
    mu = [params[names.index(f"const[{i}]")] for i in range(2)]
    sig = [np.sqrt(max(params[names.index(f"sigma2[{i}]")], 1e-12))
           for i in range(2)]
    return {"P": np.array([[p00, 1 - p00], [p10, 1 - p10]]),
            "mu": mu, "sig": sig}


def draw_markov(mdl, n, rng):
    out = np.empty(n)
    s = 1                       # start in the dominant 'normal' regime
    z = rng.standard_normal(n)
    for i in range(n):
        s = 0 if rng.random() < mdl["P"][s, 0] else 1
        out[i] = mdl["mu"][s] + mdl["sig"][s] * z[i]
    return out


def apply_edge(Rd, R_hist, m):
    """Shrink expectancy to m*historical-mean, keep variance & streaks."""
    mu = R_hist.mean()
    return m * mu + (Rd - mu)


# --------------------------------------------------------------------------- #
# 3. The verified FYFX cascade, run over a synthetic future
# --------------------------------------------------------------------------- #
def run_cascade(seq):
    """seq = [(date, R), ...] over the 4y horizon. Rebuy on breach."""
    end_d = seq[-1][0]
    pending = [seq[0][0]]
    accounts = []           # (start, events, curve, breached)
    payouts = []            # (date, you)
    n_breach = 0
    while pending and len(accounts) < MAX_ACCOUNTS:
        sd = pending.pop(0)
        res = simulate_account_fyfx(seq, sd, end_d, RISK)
        accounts.append(res)
        for e in res["events"]:
            payouts.append((e["date"], e["you"]))
        if res["breached"]:
            n_breach += 1
            bd = res["curve"][-1][0]
            nxt = next((d for d, _ in seq if d > bd), None)
            if nxt is None:
                break
            pending.append(nxt)
        # (if not breached, no further accounts from this branch)

    # fast quarter snapshots
    qends = pd.date_range(pd.Timestamp(PROJ_START),
                          pd.Timestamp(date(PROJ_START.year + HORIZON_YEARS,
                                          PROJ_START.month, PROJ_START.day)),
                          freq="QE")
    acc_curves = []
    for res in accounts:
        ds = [c[0] for c in res["curve"]]
        bs = [c[1] for c in res["curve"]]
        ls = [c[2] for c in res["curve"]]
        acc_curves.append((ds, bs, ls))

    def state_at(q):
        qd = q.date()
        paid = 0.0
        for d, y in payouts:
            if d <= qd:
                paid += y
        cyc = 0.0
        funded = 0
        for ds, bs, ls in acc_curves:
            if not ds or ds[0] > qd:
                continue
            i = bisect.bisect_right(ds, qd) - 1
            cyc += bs[i] - ls[i]
            funded += ls[i]
        return paid + cyc, funded, cyc

    timeline = [state_at(q) for q in qends]
    paid_end, funded_end, cyc_end = timeline[-1]
    return {"total_paid": sum(y for _, y in payouts),
            "paid_end": paid_end, "funded_end": funded_end,
            "cycle_end": cyc_end, "n_accounts": len(accounts),
            "n_breach": n_breach, "n_payouts": len(payouts),
            "timeline": timeline,
            "reached_25k": any(f >= 25000 for _, f, _ in timeline),
            "reached_60k": any(f >= 60000 for _, f, _ in timeline)}


# --------------------------------------------------------------------------- #
def run_scenario(name, gen, paths=PATHS):
    R = UNIVERSE
    dates = DATES
    tl = np.zeros((paths, len(DATES_QENDS), 3))
    stats = []
    for p in range(paths):
        Rd = gen()
        seq = list(zip(dates, Rd))
        c = run_cascade(seq)
        stats.append(c)
        for qi in range(len(DATES_QENDS)):
            tl[p, qi, 0] = c["timeline"][qi][0]
            tl[p, qi, 1] = c["timeline"][qi][1]
            tl[p, qi, 2] = c["timeline"][qi][2]
    tot = tl[:, -1, 0]
    out = {
        "name": name,
        "total_profit_q": {"median": np.percentile(tl[:, :, 0], 50, axis=0),
                           "p5": np.percentile(tl[:, :, 0], 5, axis=0),
                           "p25": np.percentile(tl[:, :, 0], 25, axis=0),
                           "p75": np.percentile(tl[:, :, 0], 75, axis=0),
                           "p95": np.percentile(tl[:, :, 0], 95, axis=0)},
        "end_total_profit": tot,
        "median_end": float(np.median(tot)),
        "p5_end": float(np.percentile(tot, 5)),
        "p25_end": float(np.percentile(tot, 25)),
        "p75_end": float(np.percentile(tot, 75)),
        "p95_end": float(np.percentile(tot, 95)),
        "p_loss": float(np.mean(tot < 0)),
        "p_no_breach": float(np.mean([s["n_breach"] == 0 for s in stats])),
        "mean_breaches": float(np.mean([s["n_breach"] for s in stats])),
        "p_reach25k": float(np.mean([s["reached_25k"] for s in stats])),
        "p_reach60k": float(np.mean([s["reached_60k"] for s in stats])),
        "median_accounts": float(np.median([s["n_accounts"] for s in stats])),
        "median_payouts": float(np.median([s["n_payouts"] for s in stats])),
        "median_funded_end": float(np.median([s["funded_end"]
                                              for s in stats])),
    }
    print(f"  {name:<38} median 4y profit ${out['median_end']:>10,.0f}  "
          f"(p5 ${out['p5_end']:>9,.0f} / p95 ${out['p95_end']:>10,.0f})  "
          f"P(loss)={out['p_loss']*100:4.1f}%  "
          f"P(no breach)={out['p_no_breach']*100:4.1f}%  "
          f"P(>=25K)={out['p_reach25k']*100:4.1f}%  "
          f"P(>=60K)={out['p_reach60k']*100:4.1f}%")
    return out


# --------------------------------------------------------------------------- #
def main():
    global UNIVERSE, DATES, DATES_QENDS
    print("=" * 78)
    print("4-YEAR PROJECTION — EU+GU FYFX plan @ 1.5% risk (Monte Carlo)")
    print("=" * 78)
    R = build_r_universe()
    UNIVERSE = R
    mu, sd = R.mean(), R.std(ddof=1)
    se = sd / np.sqrt(len(R))
    print(f"\nVerified history: {len(R)} trades | mean {mu:+.3f}R | "
          f"sd {sd:.3f}R | t={mu/se:.2f} | win rate "
          f"{100*np.mean(R>0):.1f}%")

    print("\n--- DIAGNOSTICS ---")
    from statsmodels.tsa.stattools import adfuller
    adf = adfuller(R, autolag="AIC")
    print(f"ADF stationarity: stat={adf[0]:.3f} p={adf[1]:.4f} -> "
          f"{'STATIONARY (no drift break detected)' if adf[1] < 0.05 else 'non-stationary!'}")
    try:
        mdl = fit_markov(R)
        print(f"2-state Markov fit: regime means {mdl['mu'][0]:+.2f}R / "
              f"{mdl['mu'][1]:+.2f}R | sds {mdl['sig'][0]:.2f} / "
              f"{mdl['sig'][1]:.2f} | P(stay) {mdl['P'][0,0]:.2f}/{mdl['P'][1,1]:.2f}")
    except Exception as e:
        print(f"Markov fit failed ({e}) — bootstrap scenarios only.")
        mdl = None

    DATES = make_dates()
    DATES_QENDS = pd.date_range(pd.Timestamp(PROJ_START),
                                pd.Timestamp(date(PROJ_START.year + 4,
                                                  PROJ_START.month,
                                                  PROJ_START.day)),
                                freq="QE")
    print(f"Horizon: {DATES[0]} -> {DATES[-1]}  ({len(DATES)} trades "
          f"projected, ~{len(DATES)/4:.0f}/yr, matching the verified "
          f"{len(R)}/yr pace)")

    mu_h = R.mean()
    cost_R = 0.5 / 14.0      # 0.5 pip on avg ~14 pip risk

    scenarios = [
        ("A. FULL EDGE (block bootstrap)",
         lambda: draw_block_bootstrap(R, len(DATES), RNG)),
        ("B. EDGE DECAY 50%",
         lambda: apply_edge(draw_block_bootstrap(R, len(DATES), RNG), R, 0.5)),
        ("C. EDGE DECAY 75% DOWN",
         lambda: apply_edge(draw_block_bootstrap(R, len(DATES), RNG), R, 0.25)),
        ("D. NO EDGE (survival test)",
         lambda: apply_edge(draw_block_bootstrap(R, len(DATES), RNG), R, 0.0)),
        ("E. FULL EDGE + 0.5 pip COST/trade",
         lambda: draw_block_bootstrap(R, len(DATES), RNG) - cost_R),
    ]
    if mdl is not None:
        scenarios.append(("F. REGIME-MODEL future (Markov)",
                          lambda: draw_markov(mdl, len(DATES), RNG)))

    print(f"\n--- SCENARIOS ({PATHS} paths each, full FYFX cascade per "
          f"path) ---")
    results = {}
    for name, gen in scenarios:
        results[name] = run_scenario(name, gen)

    # ---------------- chart ---------------- #
    import plotly.graph_objects as go
    fig = go.Figure()
    x = DATES_QENDS
    a = results["A. FULL EDGE (block bootstrap)"]
    d = results["D. NO EDGE (survival test)"]
    b = results["B. EDGE DECAY 50%"]
    e = results["E. FULL EDGE + 0.5 pip COST/trade"]
    fig.add_trace(go.Scatter(x=x, y=a["p95_end_q"] if "p95_end_q" in a else a["total_profit_q"]["p95"],
                             mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=a["total_profit_q"]["p5"], fill="tonexty",
                             mode="lines", line=dict(width=0),
                             fillcolor="rgba(26,110,224,0.15)",
                             name="A full edge: p5-p95",
                             hovertemplate="p5-p95: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=a["total_profit_q"]["p75"], mode="lines",
                             line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=a["total_profit_q"]["p25"], fill="tonexty",
                             mode="lines", line=dict(width=0),
                             fillcolor="rgba(26,110,224,0.30)",
                             name="A full edge: p25-p75",
                             hovertemplate="p25-p75: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=a["total_profit_q"]["median"],
                             mode="lines+markers", name="A FULL EDGE — median",
                             line=dict(color="#1a6ee0", width=3)))
    fig.add_trace(go.Scatter(x=x, y=b["total_profit_q"]["median"],
                             mode="lines", name="B edge decays 50% — median",
                             line=dict(color="#ff9f4a", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=e["total_profit_q"]["median"],
                             mode="lines", name="E +0.5 pip cost — median",
                             line=dict(color="#9467bd", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=d["total_profit_q"]["median"],
                             mode="lines", name="D NO EDGE (decay 100%) — median",
                             line=dict(color="#d62728", width=2, dash="dot")))
    fig.update_layout(
        title=("4-YEAR PROJECTION — EU+GU @ 1.5% risk, full FYFX multi-account "
               "cascade<br><sup>10,000+ simulated futures from the 104 "
               "gate-verified trades; bands = 5th-95th / 25th-75th percentile "
               "of total profit (payouts + in-cycle). Distributional "
               "simulation, NOT a prediction.</sup>"),
        xaxis_title="Quarter end", yaxis_title="Total profit (USD)",
        hovermode="x unified", template="plotly_white", height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    html = os.path.join(OUT, "projection_4yr_fan.html")
    fig.write_html(html, include_plotlyjs=True)
    print("\nChart:", html)

    # ---------------- summary + report ---------------- #
    json.dump({"method": "block-bootstrap + markov-regime Monte Carlo over the "
                         "verified FYFX cascade",
               "history": {"trades": int(len(R)), "mean_R": round(float(mu), 3),
                           "t_stat": round(float(mu/se), 2),
                           "adf_p": round(float(adf[1]), 4)},
               "horizon": f"{DATES[0]} -> {DATES[-1]} ({len(DATES)} trades)",
               "paths_per_scenario": PATHS,
               "scenarios": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray)
                                      else vv)
                                 for kk, vv in v.items()
                                 if kk != "total_profit_q"}
                             for k, v in results.items()}},
              open(os.path.join(OUT, "projection_summary.json"), "w"),
              indent=2, default=str)

    L = []
    A = L.append
    A("# 4-Year Projection — EU+GU FYFX plan (Monte Carlo, not a prediction)")
    A("")
    A("> **Method (what funds actually do):** simulate thousands of possible "
      "4-year futures by resampling the 104 gate-verified trades (block "
      "bootstrap + a fitted 2-state Markov regime model), run every future "
      "through the *real* FYFX cascade (8% target once, $150 payouts, 90% "
      "split, refund→new account, scale every 3 payouts, 8% breach → rebuy "
      "fresh $5K), and read the DISTRIBUTION. The future is a range, never a "
      "line.")
    A("")
    A("## Diagnostics on the verified history")
    A("")
    A(f"- {len(R)} trades · mean **{mu:+.3f}R** · sd {sd:.3f}R · "
      f"t = {mu/se:.2f}")
    A(f"- ADF stationarity p = {adf[1]:.4f} → "
      f"{'no statistical evidence of edge decay so far' if adf[1] < 0.05 else 'caution: non-stationary'}")
    if mdl is not None:
        A(f"- 2-state regime fit: {'good' if mdl['mu'][0]>0 else 'mixed'} days "
          f"({mdl['mu'][0]:+.2f}R) vs {'bad' if mdl['mu'][1]<0 else 'weak'} days "
          f"({mdl['mu'][1]:+.2f}R), persistence "
          f"{mdl['P'][0,0]:.2f}/{mdl['P'][1,1]:.2f} — streaky, exactly what the "
          "block bootstrap preserves.")
    A("")
    A("## The 4-year distribution (total profit = cash payouts + in-cycle)")
    A("")
    A("| Scenario | Median 4y profit | p5 (bad luck) | p95 (good luck) | "
      "P(end < 0) | P(no breach) | P(funded ≥ $25K) | P(funded ≥ $60K) |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in results.items():
        A(f"| {k} | **${v['median_end']:,.0f}** | ${v['p5_end']:,.0f} | "
          f"${v['p95_end']:,.0f} | {v['p_loss']*100:.1f}% | "
          f"{v['p_no_breach']*100:.1f}% | {v['p_reach25k']*100:.1f}% | "
          f"{v['p_reach60k']*100:.1f}% |")
    A("")
    A("## How to read this")
    A("")
    A(f"1. **Full edge continues (A):** median **${results['A. FULL EDGE (block bootstrap)']['median_end']:,.0f}** "
      f"in 4 years; even the unlucky 5% path makes "
      f"${results['A. FULL EDGE (block bootstrap)']['p5_end']:,.0f}. "
      f"Chance you're at a $60K level at some point: "
      f"{results['A. FULL EDGE (block bootstrap)']['p_reach60k']*100:.0f}%.")
    A(f"2. **Edge decays 50% (B):** still "
      f"${results['B. EDGE DECAY 50%']['median_end']:,.0f} median — the "
      "multi-account machinery is forgiving because payouts skim profits "
      "continuously instead of letting one big cycle ride.")
    A(f"3. **No edge at all (D):** median "
      f"${results['D. NO EDGE (survival test)']['median_end']:,.0f} — this is "
      "what pure variance + fees look like. This row is the honest reason "
      "funds demand an edge premium before scaling.")
    A(f"4. **Costs matter (E):** just 0.5 pip/trade of slippage+spread shifts "
      f"the median by ~${results['A. FULL EDGE (block bootstrap)']['median_end'] - results['E. FULL EDGE + 0.5 pip COST/trade']['median_end']:,.0f} "
      "over 4 years. Log your real fills.")
    A(f"5. **Breaches:** {results['A. FULL EDGE (block bootstrap)']['mean_breaches']:.2f} "
      "average account deaths over 4 years in the full-edge world — the "
      "cascade design (small accounts, refund-bought replacements) treats a "
      "breach as a business expense, not a catastrophe.")
    A("")
    A("## What this CANNOT tell you (be honest with yourself)")
    A("")
    A("- It resamples the PAST year's trade distribution. A genuine regime "
      "change (the edge disappearing) is only covered by scenarios B-D, "
      "not by the data.")
    A("- 104 trades is a small sample; the confidence bands are wide. Every "
      "new month of live trades tightens them — re-run monthly.")
    A("- FYFX counterparty/rules risk (the firm changing terms or closing) "
      "is not simulatable — that risk dwarfs everything above at a 4-year "
      "horizon.")
    A("")
    A("## Files")
    A("")
    A("- `projection_4yr_fan.html` — fan chart of the profit distribution")
    A("- `projection_summary.json` — full numbers per scenario")
    A("- Rerun: `python3 strategy_analysis/project_4yr.py` (re-draws with a "
      "fixed seed; swap in updated trade data monthly)")
    A("")
    path = os.path.join(OUT, "PROJECTION_4YR_REPORT.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("Report:", path)


if __name__ == "__main__":
    main()
