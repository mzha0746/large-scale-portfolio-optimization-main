"""
Markowitz Mean-Variance Optimization with three covariance estimators and
Equal Risk Contribution (Risk Parity) portfolio.

Extensions over naive MVO:
  1. Ledoit-Wolf (2004) shrinkage: reduces estimation error for large n by
     shrinking the sample covariance toward a scaled identity matrix:
         Σ_LW = (1−α)·Σ_sample + α·μ̄·I
     Optimal α is estimated analytically (Oracle Approximating Shrinkage).

  2. Risk Parity / ERC (Roncalli 2013): each asset contributes equally to
     portfolio risk.  Solved via the log-barrier convex program:
         min_{x > 0}  x'Σx − Σ_i log(x_i)   (no simplex constraint)
     Solution satisfies x_i·(Σx)_i = constant ∀i.

  3. Eigenvalue analysis: Ledoit-Wolf tightens the spectrum and reduces
     the condition number, making the optimisation better conditioned.

  4. Diversification Ratio: (Σ w_i σ_i) / σ_p — compares portfolio types.

Outputs
-------
outputs/fig1_efficient_frontier.png     (frontier comparison + RP point)
outputs/fig1b_shrinkage_analysis.png    (eigenvalue spectrum + div ratio)
outputs/markowitz_frontier_sample.csv
outputs/markowitz_frontier_lw.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from utils import DATA_DIR, OUTPUT_DIR, set_style, save_fig, ledoit_wolf, risk_parity, erc_quality

OUTPUT_DIR.mkdir(exist_ok=True)
RF = 0.04


def load_data(n_assets: int = 100):
    ret   = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    ret   = ret.iloc[:, :n_assets]
    mu    = ret.mean().values * 252
    Sigma = ret.cov().values  * 252
    return ret, mu, Sigma


def solve_mvo(mu, Sigma, r_target: float):
    n = len(mu)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)),
                      [cp.sum(x) == 1, mu @ x >= r_target, x >= 0])
    prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-8, eps_rel=1e-8)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return x.value


def efficient_frontier(mu, Sigma, n_points: int = 70):
    targets = np.linspace(mu.min() + 1e-4, mu.max() - 1e-4, n_points)
    rows = []
    for r in targets:
        w = solve_mvo(mu, Sigma, r)
        if w is not None and w.min() > -1e-4:
            pv = float(w @ Sigma @ w)
            rows.append({"return": r, "vol": np.sqrt(pv),
                         "sharpe": (r - RF) / np.sqrt(pv)})
    return pd.DataFrame(rows)


def diversification_ratio(w: np.ndarray, Sigma: np.ndarray) -> float:
    sigma_i = np.sqrt(np.diag(Sigma))
    sigma_p = np.sqrt(float(w @ Sigma @ w))
    return float(w @ sigma_i) / sigma_p


def main():
    set_style()
    print("=" * 60)
    print("Markowitz MVO  +  Ledoit-Wolf Shrinkage  +  Risk Parity")
    print("=" * 60)

    ret, mu, Sigma_s = load_data(n_assets=100)
    n = len(mu)
    T = len(ret)
    print(f"  Universe: n={n} assets,  T={T} days  (n/T={n/T:.2f})")

    # ── Ledoit-Wolf covariance ─────────────────────────────────────────────────
    print("  Fitting Ledoit-Wolf shrinkage …")
    Sigma_lw, alpha_lw = ledoit_wolf(ret)
    cn_s  = np.linalg.cond(Sigma_s)
    cn_lw = np.linalg.cond(Sigma_lw)
    print(f"  Shrinkage α = {alpha_lw:.4f}  |  "
          f"Condition number: {cn_s:.0f} → {cn_lw:.0f}  "
          f"({cn_s/cn_lw:.1f}× reduction)")

    # ── Efficient frontiers ────────────────────────────────────────────────────
    print("  Computing sample MVO frontier …")
    front_s  = efficient_frontier(mu, Sigma_s,  n_points=80)
    print("  Computing Ledoit-Wolf frontier …")
    front_lw = efficient_frontier(mu, Sigma_lw, n_points=80)

    front_s.to_csv(OUTPUT_DIR  / "markowitz_frontier_sample.csv", index=False)
    front_lw.to_csv(OUTPUT_DIR / "markowitz_frontier_lw.csv",    index=False)

    msr_s  = front_s.iloc[front_s["sharpe"].idxmax()]
    msr_l  = front_lw.iloc[front_lw["sharpe"].idxmax()]
    mvp_s  = front_s.iloc[front_s["vol"].idxmin()]
    mvp_l  = front_lw.iloc[front_lw["vol"].idxmin()]

    print(f"\n  {'':5} {'Max-Sharpe':>30}  {'Min-Var':>22}")
    for label, msr, mvp in [("Sample", msr_s, mvp_s), ("LW    ", msr_l, mvp_l)]:
        print(f"  {label}  μ={msr['return']:.2%} σ={msr['vol']:.2%} SR={msr['sharpe']:.2f}"
              f"   |   μ={mvp['return']:.2%} σ={mvp['vol']:.2%}")

    # ── Risk Parity (ERC) ──────────────────────────────────────────────────────
    print("\n  Computing Risk Parity (log-barrier ERC) …")
    w_rp_s  = risk_parity(Sigma_s)
    w_rp_lw = risk_parity(Sigma_lw)

    for label, w, Sig in [("Sample", w_rp_s, Sigma_s), ("LW    ", w_rp_lw, Sigma_lw)]:
        r   = float(mu @ w)
        vol = float(np.sqrt(w @ Sig @ w))
        q   = erc_quality(w, Sig) * 1000
        dr  = diversification_ratio(w, Sig)
        print(f"  RP ({label}):  μ={r:.2%}  σ={vol:.2%}  "
              f"SR={(r-RF)/vol:.2f}  RC_std={q:.4f}‰  DivRatio={dr:.3f}")

    # ── Diversification Ratio comparison ──────────────────────────────────────
    w_ew   = np.ones(n) / n
    w_msr_s = solve_mvo(mu, Sigma_s, float(msr_s["return"]))
    if w_msr_s is None:
        w_msr_s = w_ew
    w_msr_s = np.clip(w_msr_s, 0, None); w_msr_s /= w_msr_s.sum()

    strats = {
        "1/N EW":          (w_ew,    Sigma_s),
        "Max-SR (sample)": (w_msr_s, Sigma_s),
        "Risk Parity (LW)":(w_rp_lw, Sigma_lw),
    }
    print("\n  Diversification Ratios:")
    for label, (w, Sig) in strats.items():
        print(f"    {label:<22}: {diversification_ratio(w, Sig):.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 1: Efficient frontiers + RP point
    # ─────────────────────────────────────────────────────────────────────────
    asset_vol = np.sqrt(np.diag(Sigma_s))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(asset_vol, mu, s=14, alpha=0.35, color="#94a3b8",
               label="Individual assets", zorder=1)
    ax.plot(front_s["vol"],  front_s["return"],  color="#2563eb", lw=2,
            label="Sample MVO")
    ax.plot(front_lw["vol"], front_lw["return"], color="#dc2626", lw=2, ls="--",
            label="Ledoit-Wolf MVO")
    ax.scatter(msr_s["vol"], msr_s["return"], marker="*", s=220,
               color="#2563eb", zorder=5,
               label=f"Max-SR (sample, SR={msr_s['sharpe']:.2f})")
    ax.scatter(msr_l["vol"], msr_l["return"], marker="*", s=220,
               color="#dc2626", zorder=5,
               label=f"Max-SR (LW,     SR={msr_l['sharpe']:.2f})")
    rp_ret = float(mu @ w_rp_lw)
    rp_vol = float(np.sqrt(w_rp_lw @ Sigma_lw @ w_rp_lw))
    ax.scatter(rp_vol, rp_ret, marker="D", s=140, color="#16a34a", zorder=6,
               label=f"Risk Parity (LW)  SR={(rp_ret-RF)/rp_vol:.2f}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set(xlabel="Annualised Volatility", ylabel="Annualised Return",
           title=f"Efficient Frontiers: Sample vs. Ledoit-Wolf (n={n})")
    ax.legend(fontsize=8)

    # Right: ERC verification — risk contributions should all equal 1/n
    ax2 = axes[1]
    RC = w_rp_lw * (Sigma_lw @ w_rp_lw) / float(w_rp_lw @ Sigma_lw @ w_rp_lw)
    ax2.bar(range(n), np.sort(RC)[::-1] * 100,
            color="#2563eb", alpha=0.75, width=0.9, label="Risk Contribution (%)")
    ax2.bar(range(n), np.sort(w_rp_lw)[::-1] * 100,
            color="#dc2626", alpha=0.45, width=0.9, label="Portfolio Weight (%)")
    ax2.axhline(100 / n, color="#16a34a", ls="--", lw=2,
                label=f"ERC target = {100/n:.2f}%")
    ax2.set(xlabel="Asset (sorted by weight)", ylabel="%",
            title="Risk Parity: Risk Contribution ≈ Portfolio Weight (LW covariance)")
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"Mean-Variance Analysis  |  n={n} assets  |  LW shrinkage α={alpha_lw:.3f}",
        fontsize=11)
    save_fig("fig1_efficient_frontier")

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 1b: Ledoit-Wolf shrinkage effect analysis
    # ─────────────────────────────────────────────────────────────────────────
    eig_s  = np.sort(np.linalg.eigvalsh(Sigma_s))[::-1]
    eig_lw = np.sort(np.linalg.eigvalsh(Sigma_lw))[::-1]

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: eigenvalue spectrum
    ax3 = axes2[0]
    ax3.semilogy(range(1, n+1), eig_s,  color="#2563eb", lw=1.5,
                 label=f"Sample (cond={cn_s:.0f})")
    ax3.semilogy(range(1, n+1), eig_lw, color="#dc2626", lw=1.5, ls="--",
                 label=f"LW (cond={cn_lw:.0f})")
    ax3.set(xlabel="Eigenvalue rank", ylabel="Eigenvalue (log scale)",
            title="Covariance Spectrum: Shrinkage Reduces\nIll-Conditioning")
    ax3.legend(fontsize=9)

    # Middle: effective N (portfolio concentration measure)
    # Effective number of bets = 1 / Σ_i (x_i)^2  (inverse Herfindahl)
    def eff_n(w): return 1.0 / np.sum(w**2)
    labs = ["1/N EW", "Max-SR\n(sample)", "Min-Var\n(LW)", "Risk Parity\n(LW)"]
    w_mvp_lw = solve_mvo(mu, Sigma_lw, float(mvp_l["return"]))
    if w_mvp_lw is None: w_mvp_lw = w_ew
    w_mvp_lw = np.clip(w_mvp_lw, 0, None); w_mvp_lw /= w_mvp_lw.sum()
    ens = [eff_n(w_ew), eff_n(w_msr_s), eff_n(w_mvp_lw), eff_n(w_rp_lw)]
    bar_colors = ["#94a3b8", "#2563eb", "#9333ea", "#16a34a"]
    ax4 = axes2[1]
    ax4.bar(labs, ens, color=bar_colors, alpha=0.85, width=0.55)
    ax4.axhline(n, color="#94a3b8", ls="--", lw=1.5, label=f"Max = {n}")
    for i, v in enumerate(ens):
        ax4.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax4.set(ylabel="Effective Number of Assets (1/HHI)",
            title="Portfolio Concentration\n(higher = more diversified)")
    ax4.legend(fontsize=9)

    # Right: Sharpe ratio at different target returns (sample vs. LW)
    ax5 = axes2[2]
    ax5.plot(front_s["vol"],  front_s["sharpe"],  color="#2563eb", lw=2,
             label="Sample MVO")
    ax5.plot(front_lw["vol"], front_lw["sharpe"], color="#dc2626", lw=2, ls="--",
             label="LW MVO")
    ax5.scatter(rp_vol, (rp_ret - RF) / rp_vol, marker="D", s=100,
                color="#16a34a", zorder=5, label="Risk Parity (LW)")
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax5.set(xlabel="Portfolio Volatility", ylabel="Sharpe Ratio",
            title="Sharpe Ratio across the Frontier")
    ax5.legend(fontsize=9)

    fig2.suptitle(
        "Ledoit-Wolf Shrinkage Analysis  |  ERC Diversification Metrics",
        fontsize=11)
    save_fig("fig1b_shrinkage_analysis")

    print("\n  Done.")
    return front_s, front_lw, w_rp_lw


if __name__ == "__main__":
    main()
