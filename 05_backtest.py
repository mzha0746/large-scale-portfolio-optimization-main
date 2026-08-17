"""
Rolling-window out-of-sample backtest with transaction costs.

Strategies compared (all long-only):
  1. Equal-Weight (1/N)            — naive benchmark
  2. MVO Max-Sharpe (sample Σ)     — classical Markowitz
  3. MVO Max-Sharpe (Ledoit-Wolf Σ)— shrinkage-enhanced Markowitz
  4. Risk Parity / ERC (LW Σ)      — equal risk contribution
  5. Cardinality K=20 (MIQP/topK)  — sparse allocation
  6. Lagrangian K=20               — approximate CCMVP

Transaction cost model:
  Each rebalancing incurs proportional cost C_TX = 10bps per unit of turnover:
      TC = C_TX · Σ_i |w_new_i − w_prev_i|
  The cost is deducted from the first day of each new holding period.

Rolling-window design:
  - Estimate μ, Σ from trailing TRAIN_MONTHS of daily data
  - Rebalance every REBAL_MONTHS, hold until next rebalancing date
  - Record daily returns net of transaction costs
  - All Sharpe / MDD / turnover statistics computed on the daily series

Outputs
-------
outputs/fig4_backtest_wealth.png     (cumulative wealth paths)
outputs/fig4b_rolling_sharpe.png     (12-month rolling Sharpe)
outputs/fig4c_turnover.png           (annual turnover by strategy)
outputs/backtest_summary.csv
"""
import time, warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import (DATA_DIR, OUTPUT_DIR, set_style, save_fig,
                   sharpe, max_drawdown, ledoit_wolf, risk_parity)

OUTPUT_DIR.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

TRAIN_MONTHS = 24
REBAL_MONTHS = 1
K_CARD       = 20
R_FLOOR      = 0.04
RF_ANNUAL    = 0.04
C_TX         = 0.001    # 10 bps one-way transaction cost


# ── strategy solvers ───────────────────────────────────────────────────────────

def ew_weights(n): return np.ones(n) / n


def mvo_max_sharpe(mu, Sigma):
    """Max-Sharpe via QP sweep over target returns."""
    n = len(mu)
    r_grid = np.linspace(mu.min() + 1e-4, mu.max() - 1e-4, 30)
    best_sr, best_w = -np.inf, np.ones(n) / n
    for r in r_grid:
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)),
                          [mu @ x >= r, cp.sum(x) == 1, x >= 0])
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
            w = np.clip(x.value, 0, None); w /= w.sum()
            vol = np.sqrt(float(w @ Sigma @ w))
            sr  = (float(mu @ w) - RF_ANNUAL) / vol if vol > 1e-12 else 0.0
            if sr > best_sr:
                best_sr, best_w = sr, w
    return best_w


def cardinality_topk(mu, Sigma, K, r_target):
    """Top-K CVXPY fallback (MVO → select top-K by weight → restricted QP)."""
    n = len(mu)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)),
                      [mu @ x >= r_target, cp.sum(x) == 1, x >= 0])
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        return np.ones(n) / n
    w = np.clip(x.value, 0, None)
    idx = np.argsort(w)[::-1][:K]
    xk = cp.Variable(K, nonneg=True)
    prob2 = cp.Problem(cp.Minimize(cp.quad_form(xk, Sigma[np.ix_(idx, idx)])),
                       [mu[idx] @ xk >= r_target, cp.sum(xk) == 1])
    prob2.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob2.status in ("optimal", "optimal_inaccurate") and xk.value is not None:
        wf = np.zeros(n); wf[idx] = np.clip(xk.value, 0, None)
        return wf / wf.sum()
    return np.ones(n) / n


def cardinality_miqp(mu, Sigma, K, r_target):
    try:
        import gurobipy as gp
        from gurobipy import GRB
        n = len(mu)
        m = gp.Model(); m.setParam("OutputFlag", 0); m.setParam("TimeLimit", 30)
        x = m.addMVar(n, lb=0, ub=1); z = m.addMVar(n, vtype=GRB.BINARY)
        m.setObjective(x @ Sigma @ x, GRB.MINIMIZE)
        m.addConstr(mu @ x >= r_target); m.addConstr(x.sum() == 1)
        m.addConstr(x <= z); m.addConstr(z.sum() <= K)
        m.optimize()
        if m.Status in (2, 9):
            return x.X / x.X.sum()
    except Exception:
        pass
    return cardinality_topk(mu, Sigma, K, r_target)


def lagrangian_fast(mu, Sigma, K, r_target, max_iter=60):
    """Truncated Lagrangian relaxation for rolling backtest."""
    n = len(mu); lam = np.zeros(n); best_w = np.ones(n)/n; best_ub = np.inf
    for _ in range(max_iter):
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma) + lam @ x),
                          [mu @ x >= r_target, cp.sum(x) == 1, x >= 0])
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"): break
        xv = np.clip(x.value, 0, None)
        idx = np.argsort(lam)[::-1][:K]
        zv  = np.zeros(n); zv[idx] = 1
        xr  = cp.Variable(K, nonneg=True)
        pr2 = cp.Problem(cp.Minimize(cp.quad_form(xr, Sigma[np.ix_(idx, idx)])),
                         [mu[idx] @ xr >= r_target, cp.sum(xr) == 1])
        pr2.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if pr2.status in ("optimal", "optimal_inaccurate") and pr2.value is not None:
            wf = np.zeros(n); wf[idx] = np.clip(xr.value, 0, None)
            if float(wf @ Sigma @ wf) < best_ub:
                best_ub = float(wf @ Sigma @ wf); best_w = wf
        lb = float(xv @ Sigma @ xv) + float(lam @ (xv - zv))
        sg = xv - zv
        if np.dot(sg, sg) < 1e-14: break
        alpha = max((best_ub - lb) / np.dot(sg, sg), 0)
        lam   = np.maximum(0, lam + alpha * sg)
    s = best_w.sum()
    return best_w / s if s > 0 else np.ones(n) / n


# ── rolling backtest engine ────────────────────────────────────────────────────

def main():
    set_style()
    print("=" * 60)
    print("Rolling Backtest — 6 strategies, transaction costs, daily P&L")
    print("=" * 60)

    ret = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    ret = ret.iloc[:, :50]       # 50 assets for speed
    n   = ret.shape[1]

    monthly     = ret.resample("ME").last().index
    test_dates  = monthly[TRAIN_MONTHS:]
    strat_names = ["EW", "MVO-Sample", "MVO-LW", "RiskParity", "Card-K20", "Lagr-K20"]

    # daily return series per strategy (keyed by name)
    daily_rets_all : dict[str, list] = {s: [] for s in strat_names}
    dates_by_strat : dict[str, list] = {s: [] for s in strat_names}
    turnover_all   : dict[str, list] = {s: [] for s in strat_names}

    # previous weights for transaction cost computation
    w_prev = {s: np.ones(n) / n for s in strat_names}

    print(f"  Assets={n}  Train={TRAIN_MONTHS}m  C_TX={C_TX*1e4:.0f}bps  "
          f"Rebal periods={len(test_dates)-1}")

    for i, t_end in enumerate(test_dates[:-1]):
        t_start_train = monthly[i]
        t_start_test  = t_end
        t_end_test    = test_dates[i + 1]

        train = ret.loc[t_start_train:t_end]
        test  = ret.loc[t_start_test:t_end_test]
        if len(train) < 60 or len(test) == 0:
            continue

        mu_t    = train.mean().values * 252
        Sig_s   = train.cov().values  * 252
        try:
            Sig_lw, _ = ledoit_wolf(train)
        except Exception:
            Sig_lw = Sig_s

        r_tgt = max(float(np.percentile(mu_t, 55)), R_FLOOR)

        # ── solve each strategy ────────────────────────────────────────────────
        portfolios = {
            "EW":         ew_weights(n),
            "MVO-Sample": mvo_max_sharpe(mu_t, Sig_s),
            "MVO-LW":     mvo_max_sharpe(mu_t, Sig_lw),
            "RiskParity": risk_parity(Sig_lw),
            "Card-K20":   cardinality_miqp(mu_t, Sig_s, K=K_CARD, r_target=r_tgt),
            "Lagr-K20":   lagrangian_fast(mu_t, Sig_s,  K=K_CARD, r_target=r_tgt),
        }

        test_vals  = test.values        # (T_test, n)
        test_dates_period = test.index

        for sname, w in portfolios.items():
            w = np.clip(np.asarray(w), 0, None)
            if w.sum() < 1e-12: w = np.ones(n) / n
            w /= w.sum()

            # transaction cost at rebalancing (deducted from first day)
            tc_total = C_TX * np.abs(w - w_prev[sname]).sum()
            turnover_all[sname].append(tc_total)
            w_prev[sname] = w.copy()

            # daily returns during holding period
            dr = test_vals @ w
            if len(dr) > 0:
                dr_net = dr.copy()
                dr_net[0] -= tc_total   # deduct full TC on day-1
                daily_rets_all[sname].extend(dr_net.tolist())
                dates_by_strat[sname].extend(test_dates_period.tolist())

        if (i + 1) % 6 == 0:
            print(f"  Rebal {i+1:2d}/{len(test_dates)-1}  "
                  f"[{t_start_train.date()} – {t_end_test.date()}]")

    # ── performance summary ────────────────────────────────────────────────────
    print("\n  === Backtest Summary ===")
    summary  = []
    cum_dict = {}
    colors   = ["#94a3b8", "#2563eb", "#6366f1", "#16a34a", "#dc2626", "#ea580c"]

    for sname in strat_names:
        r   = np.array(daily_rets_all[sname])
        if len(r) == 0:
            continue
        cum   = (1 + r).cumprod()
        sr    = sharpe(r, rf=RF_ANNUAL / 252)
        mdd   = max_drawdown(cum)
        ann   = float(cum[-1] ** (252 / max(len(r), 1)) - 1)
        vol   = float(r.std(ddof=1) * np.sqrt(252))
        to    = float(np.mean(turnover_all[sname])) * 12  # annualised (monthly × 12)
        print(f"  {sname:<12}  Ann={ann:.2%}  Vol={vol:.2%}  SR={sr:.2f}  "
              f"MDD={mdd:.2%}  Turnover={to:.1%}/yr")
        summary.append({"Strategy": sname, "Ann.Return": round(ann, 4),
                        "Volatility": round(vol, 4), "Sharpe": round(sr, 3),
                        "MaxDrawdown": round(mdd, 4),
                        "Annualised Turnover": round(to, 4)})
        cum_dict[sname] = cum

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)

    # ── Fig 4: Cumulative wealth paths ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for (sname, cum), color in zip(cum_dict.items(), colors):
        dates_plot = np.array(dates_by_strat[sname])[:len(cum)]
        ax.plot(dates_plot, cum, label=sname, color=color, lw=1.8)
    ax.axhline(1, color="#94a3b8", lw=0.8, ls="--")
    ax.set(xlabel="Date", ylabel="Cumulative Wealth (start = 1)",
           title=f"Out-of-Sample Wealth — {n} assets, 10bps TC, {TRAIN_MONTHS}m estimation window")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    save_fig("fig4_backtest_wealth")

    # ── Fig 4b: 12-month rolling Sharpe ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ROLL = 252
    for (sname, cum), color in zip(cum_dict.items(), colors):
        r = np.array(daily_rets_all[sname])
        dates_s = np.array(dates_by_strat[sname])
        if len(r) >= ROLL:
            roll_sr = pd.Series(r).rolling(ROLL).apply(
                lambda x: (x.mean() * 252 - RF_ANNUAL) / (x.std(ddof=1) * np.sqrt(252))
                if x.std(ddof=1) > 1e-12 else 0, raw=True)
            ax2.plot(dates_s[:len(roll_sr)], roll_sr.values,
                     label=sname, color=color, lw=1.5)
    ax2.axhline(0, color="#94a3b8", lw=0.8, ls="--")
    ax2.set(xlabel="Date", ylabel="12-Month Rolling Sharpe Ratio",
            title="Rolling Sharpe Ratio (252-day window)")
    ax2.legend(fontsize=8)
    save_fig("fig4b_rolling_sharpe")

    # ── Fig 4c: Annual turnover by strategy ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    to_vals  = [float(np.mean(turnover_all[s])) * 12 for s in strat_names]
    bar_cols = colors
    ax3.bar(strat_names, [v * 100 for v in to_vals], color=bar_cols, alpha=0.8)
    ax3.set(xlabel="Strategy", ylabel="Annualised Turnover (%)",
            title=f"Annual Portfolio Turnover (C_TX = {C_TX*1e4:.0f}bps)\n"
                  "Sparse strategies rebalance less aggressively")
    for i, v in enumerate(to_vals):
        ax3.text(i, v * 100 + 0.3, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
    save_fig("fig4c_turnover")

    print("\n  Done. See outputs/backtest_summary.csv")


if __name__ == "__main__":
    main()
