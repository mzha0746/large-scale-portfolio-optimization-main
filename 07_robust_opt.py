"""
Robust Cardinality-Constrained Portfolio via Ellipsoidal Uncertainty Sets.

Motivation
----------
Classical MVO assumes μ is known exactly. In practice, mean return estimates
have high estimation error (σ(μ̂) ~ σ_i / √T). Ignoring this leads to
'error-maximising' portfolios that over-concentrate in assets with high
estimated returns but large estimation uncertainty.

Robust counterpart (Lobo & Boyd 2000, Ben-Tal & Nemirovski 1998):
    Assume μ_true ∈ {μ̂ + Σ^{1/2} δ : ||δ||₂ ≤ Γ}
    where Γ controls the conservatism (uncertainty budget).

    Robust return constraint:
        min_{||δ||₂ ≤ Γ} (μ̂ + Σ^{1/2}δ)'x  =  μ̂'x − Γ·||Σ^{1/2}x||₂
                                               =  μ̂'x − Γ·√(x'Σx)
                                               =  μ̂'x − Γ·σ_p(x)

    So the robust CCMVP becomes:
        min   x'Σx                               (min variance)
        s.t.  μ̂'x − Γ·||L x||₂ ≥ r_target      (SOCP constraint)
              1'x = 1,  x ≥ 0
              x_i ≤ z_i,  Σ z_i ≤ K,  z ∈ {0,1}
    where L = chol(Σ) so ||Lx||₂ = σ_p(x).

Interpretation: larger Γ → must over-compensate expected return to ensure
constraint holds under worst-case μ → forces lower σ_p → more conservative.

Algorithm
---------
Lagrangian relaxation with robust x-subproblem (SOCP via CLARABEL):
    x-subproblem: min x'Σx + λ'x  s.t.  μ'x − Γ·||Lx|| ≥ r,  1'x=1,  x≥0
    z-subproblem: pick top-K indices of λ  (same greedy rule as before)

Γ-Sensitivity Analysis:
    Sweep Γ ∈ {0, 0.5, 1, 2, 3, 5, 10} and compare:
      - Portfolio volatility, expected return, Sharpe Ratio
      - Number of effectively selected assets
      - Out-of-sample performance on held-out test period

Outputs
-------
outputs/fig7_robust_gamma_sensitivity.png
outputs/fig7b_robust_vs_nominal.png
outputs/robust_results.csv
"""
import time, json
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import DATA_DIR, OUTPUT_DIR, set_style, save_fig, sharpe, max_drawdown

OUTPUT_DIR.mkdir(exist_ok=True)

K          = 20       # cardinality budget
MAX_ITER   = 200      # Lagrangian iterations
TOL_GAP    = 1e-4     # convergence tolerance
RF         = 0.04     # risk-free rate


def chol_safe(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky with small ridge for PSD guarantee."""
    n = Sigma.shape[0]
    for eps in [0, 1e-8, 1e-6, 1e-4]:
        try:
            return cholesky(Sigma + eps * np.eye(n), lower=True)
        except Exception:
            continue
    return np.diag(np.sqrt(np.abs(np.diag(Sigma))))


def solve_x_robust(mu, Sigma, L, lam, r_target, gamma):
    """
    Robust x-subproblem (SOCP):
        min  x'Σx + λ'x
        s.t. μ'x − γ·||Lx||₂ ≥ r_target
             1'x = 1,  x ≥ 0
    When γ = 0 this reduces to the standard QP x-subproblem.
    """
    n = len(mu)
    x = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)   # t = ||Lx||₂  (portfolio vol)

    constraints = [
        cp.sum(x) == 1,
        cp.norm(L @ x, 2) <= t,             # SOC: t = σ_p(x)
        mu @ x - gamma * t >= r_target,     # robust return constraint
    ]
    obj = cp.quad_form(x, Sigma) + lam @ x
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
        return np.clip(x.value, 0, None)
    # fallback: nominal (γ=0) QP
    x2 = cp.Variable(n, nonneg=True)
    prob2 = cp.Problem(cp.Minimize(cp.quad_form(x2, Sigma) + lam @ x2),
                       [cp.sum(x2) == 1, mu @ x2 >= r_target])
    prob2.solve(solver=cp.OSQP, verbose=False)
    return np.clip(x2.value, 0, None) if x2.value is not None else np.ones(n)/n


def feasible_qp(mu, Sigma, idx, r_target):
    """Min-variance QP on support idx (primal feasibility recovery)."""
    m = len(idx)
    xk = cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(xk, Sigma[np.ix_(idx, idx)])),
                      [mu[idx] @ xk >= r_target, cp.sum(xk) == 1])
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob.status in ("optimal", "optimal_inaccurate") and xk.value is not None:
        n = len(mu)
        w = np.zeros(n); w[idx] = np.clip(xk.value, 0, None)
        return w
    return None


def robust_lagrangian(mu, Sigma, r_target, K, gamma):
    """
    Lagrangian relaxation for robust CCMVP with uncertainty budget γ.
    Returns (best_w, UB, LB, history_df).
    """
    n  = len(mu)
    L  = chol_safe(Sigma)
    lam = np.zeros(n)
    UB, LB = np.inf, -np.inf
    best_w = np.ones(n) / n
    history = []

    for t in range(1, MAX_ITER + 1):
        # x-subproblem (SOCP with robust constraint)
        xv = solve_x_robust(mu, Sigma, L, lam, r_target, gamma)

        # z-subproblem: select K assets with largest λ_i
        idx = np.argsort(lam)[::-1][:K]
        zv  = np.zeros(n); zv[idx] = 1.0

        # Lagrangian dual lower bound
        lb = float(xv @ Sigma @ xv) + float(lam @ (xv - zv))
        LB = max(LB, lb)

        # Primal upper bound (feasible restricted QP)
        wf = feasible_qp(mu, Sigma, idx, r_target)
        if wf is not None:
            ub = float(wf @ Sigma @ wf)
            if ub < UB:
                UB = ub; best_w = wf

        gap = (UB - LB) / (abs(UB) + 1e-12)
        history.append({"iter": t, "LB": LB, "UB": UB, "gap": gap})

        if gap < TOL_GAP:
            break

        # Polyak subgradient step
        sg    = xv - zv
        sg_sq = float(sg @ sg)
        if sg_sq < 1e-14:
            break
        alpha = (UB - LB) / sg_sq
        lam   = np.maximum(0, lam + alpha * sg)

    return best_w, UB, LB, pd.DataFrame(history)


def main():
    set_style()
    print("=" * 60)
    print("Robust Cardinality Portfolio — Ellipsoidal Uncertainty")
    print("=" * 60)

    ret = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    ret = ret.iloc[:, :100]
    T, n = ret.shape

    mu    = ret.mean().values * 252
    Sigma = ret.cov().values  * 252

    r_target = float(np.percentile(mu, 60))
    print(f"  n={n} assets,  T={T} days,  K={K},  r_target={r_target:.2%}")

    # train/test split for out-of-sample evaluation
    split   = int(T * 0.70)
    ret_tr  = ret.iloc[:split]
    ret_te  = ret.iloc[split:]
    mu_tr   = ret_tr.mean().values * 252
    Sig_tr  = ret_tr.cov().values  * 252
    r_tgt_tr = float(np.percentile(mu_tr, 60))

    # Γ sensitivity sweep
    gammas  = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    records = []

    print(f"\n  {'Γ':>5}  {'port_ret':>9}  {'port_vol':>9}  {'SR':>6}  "
          f"{'n_sel':>5}  {'gap':>8}  {'iters':>6}  {'OOS_SR':>8}")
    print("  " + "-" * 68)

    for gamma in gammas:
        t0 = time.perf_counter()
        w, UB, LB, hist = robust_lagrangian(mu_tr, Sig_tr, r_tgt_tr, K, gamma)
        elapsed = time.perf_counter() - t0

        port_ret = float(mu_tr @ w)
        port_vol = float(np.sqrt(w @ Sig_tr @ w))
        sr_in    = (port_ret - RF) / port_vol if port_vol > 0 else 0.0
        n_sel    = int((w > 1e-4).sum())
        gap      = hist["gap"].iloc[-1]

        # out-of-sample evaluation
        oos_r   = ret_te.values @ w
        sr_oos  = sharpe(oos_r, rf=RF / 252)

        print(f"  {gamma:>5.1f}  {port_ret:>9.2%}  {port_vol:>9.2%}  "
              f"{sr_in:>6.2f}  {n_sel:>5d}  {gap:>8.4%}  {len(hist):>6d}  {sr_oos:>8.2f}")

        records.append({
            "gamma": gamma, "port_ret": port_ret, "port_vol": port_vol,
            "sharpe_IS": sr_in, "n_selected": n_sel, "gap": gap,
            "iters": len(hist), "sharpe_OOS": sr_oos, "solve_sec": round(elapsed, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "robust_results.csv", index=False)

    # ── Fig 7: Γ-sensitivity analysis ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-left: port_vol vs Γ
    axes[0, 0].plot(df["gamma"], df["port_vol"] * 100, "o-",
                    color="#2563eb", lw=2, ms=8)
    axes[0, 0].set(xlabel="Robustness Budget Γ",
                   ylabel="Portfolio Volatility (%)",
                   title="Robustness ↑  →  Volatility ↓\n(more conservative allocations)")

    # Top-right: port_ret and SR_IS vs Γ
    axes[0, 1].plot(df["gamma"], df["port_ret"] * 100, "s-",
                    color="#dc2626", lw=2, ms=8, label="Expected Return (%)")
    ax_twin = axes[0, 1].twinx()
    ax_twin.plot(df["gamma"], df["sharpe_IS"], "^--",
                 color="#9333ea", lw=1.5, ms=7, label="Sharpe (in-sample)")
    ax_twin.set_ylabel("Sharpe Ratio (in-sample)", color="#9333ea")
    axes[0, 1].set(xlabel="Γ", ylabel="Ann. Expected Return (%)",
                   title="Return & Sharpe vs. Robustness Budget")
    axes[0, 1].legend(loc="upper left", fontsize=8)
    ax_twin.legend(loc="upper right", fontsize=8)

    # Bottom-left: n_selected vs Γ (robustness → sparsity)
    axes[1, 0].bar(df["gamma"].astype(str), df["n_selected"],
                   color="#16a34a", alpha=0.8)
    axes[1, 0].set(xlabel="Γ", ylabel="# Assets Selected",
                   title=f"Cardinality K={K}: Robust Portfolio Sparsity")
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1, 0].text(i, row["n_selected"] + 0.15, str(int(row["n_selected"])),
                        ha="center", fontsize=9, fontweight="bold")

    # Bottom-right: OOS Sharpe vs Γ (out-of-sample benefit of robustness)
    axes[1, 1].plot(df["gamma"], df["sharpe_OOS"], "D-",
                    color="#ea580c", lw=2, ms=9)
    axes[1, 1].axvline(df.loc[df["sharpe_OOS"].idxmax(), "gamma"],
                       color="#94a3b8", ls="--", lw=1.5,
                       label=f"Best OOS Γ = {df.loc[df['sharpe_OOS'].idxmax(),'gamma']:.1f}")
    axes[1, 1].set(xlabel="Γ", ylabel="Out-of-Sample Sharpe Ratio",
                   title="Out-of-Sample Performance\n(optimal Γ balances in/out bias)")
    axes[1, 1].legend(fontsize=9)

    fig.suptitle(
        f"Robust CCMVP: Ellipsoidal Uncertainty  |  n={n}, K={K},  "
        f"r_target={r_tgt_tr:.1%}",
        fontsize=11)
    save_fig("fig7_robust_gamma_sensitivity")

    # ── Fig 7b: Nominal vs. Robust weight comparison ─────────────────────────
    w_nom  = df.loc[df["gamma"] == 0.0].index
    w_rob  = df.loc[df["sharpe_OOS"].idxmax()].name
    best_g = df.loc[df["sharpe_OOS"].idxmax(), "gamma"]

    # Re-run to get weights for nominal and best-Γ
    w_nominal, *_ = robust_lagrangian(mu_tr, Sig_tr, r_tgt_tr, K, 0.0)
    w_robust,  *_ = robust_lagrangian(mu_tr, Sig_tr, r_tgt_tr, K, best_g)

    oos_nom = ret_te.values @ w_nominal
    oos_rob = ret_te.values @ w_robust
    cum_nom = (1 + oos_nom).cumprod()
    cum_rob = (1 + oos_rob).cumprod()
    dates_oos = ret_te.index

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: OOS wealth paths
    axes2[0].plot(dates_oos[:len(cum_nom)], cum_nom,
                  color="#2563eb", lw=2, label=f"Nominal (Γ=0)  SR={sharpe(oos_nom, rf=RF/252):.2f}")
    axes2[0].plot(dates_oos[:len(cum_rob)], cum_rob,
                  color="#dc2626", lw=2, label=f"Robust  (Γ={best_g:.1f})  SR={sharpe(oos_rob, rf=RF/252):.2f}")
    axes2[0].axhline(1, color="#94a3b8", ls="--", lw=0.8)
    axes2[0].set(xlabel="Date", ylabel="Cumulative Wealth",
                 title="Out-of-Sample: Nominal vs. Robust Portfolio")
    axes2[0].legend(fontsize=9)
    axes2[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}×"))

    # Right: Weight comparison (bar chart, top 25 assets)
    all_idx = np.union1d(np.where(w_nominal > 1e-4)[0], np.where(w_robust > 1e-4)[0])
    top25   = all_idx[np.argsort(np.maximum(w_nominal[all_idx], w_robust[all_idx]))[::-1][:25]]
    labels  = [ret.columns[i] for i in top25]
    xpos    = np.arange(len(top25))
    width   = 0.38
    axes2[1].bar(xpos - width/2, w_nominal[top25] * 100, width,
                 color="#2563eb", alpha=0.85, label="Nominal (Γ=0)")
    axes2[1].bar(xpos + width/2, w_robust[top25]  * 100, width,
                 color="#dc2626", alpha=0.85, label=f"Robust (Γ={best_g:.1f})")
    axes2[1].set_xticks(xpos)
    axes2[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes2[1].set(ylabel="Weight (%)",
                 title="Portfolio Weights: Robust vs. Nominal\n"
                       "(Robust spreads allocation more evenly)")
    axes2[1].legend(fontsize=9)

    fig2.suptitle("Robust Cardinality Portfolio: Out-of-Sample Analysis", fontsize=11)
    save_fig("fig7b_robust_vs_nominal")

    print("\n  Done. See outputs/robust_results.csv")
    return df


if __name__ == "__main__":
    main()
