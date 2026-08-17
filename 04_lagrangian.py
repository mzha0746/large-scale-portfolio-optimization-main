"""
Lagrangian Relaxation for Cardinality-Constrained Portfolio Optimization.

Decomposition
-------------
The linking constraints  x_i ≤ z_i  are relaxed with multipliers λ_i ≥ 0:

    L(λ) = min_{x,z}  x'Σx + λ'(x - z)
            s.t.  μ'x ≥ r,  1'x = 1,  x ≥ 0,  Σz_i ≤ K,  z ∈ {0,1}^n

This decouples into:
  [x-subproblem]  min  x'Σx + λ'x   s.t.  μ'x ≥ r, 1'x=1, x≥0   → QP (cvxpy)
  [z-subproblem]  min  -λ'z          s.t.  Σz_i ≤ K, z∈{0,1}^n   → pick K largest λ_i

Multipliers updated via subgradient:
    λ_{t+1} = max(0,  λ_t + α_t * (x_t - z_t))
    α_t = (UB - LB) / ||x_t - z_t||²    (Polyak step)

A feasible upper bound is recovered by projecting onto the selected z support.

Outputs
-------
outputs/fig3_lagrangian_convergence.png
outputs/lagrangian_results.json
"""
import time, json
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import DATA_DIR, OUTPUT_DIR, set_style, save_fig, save_json

OUTPUT_DIR.mkdir(exist_ok=True)


# ── x-subproblem (QP) ─────────────────────────────────────────────────────────

def solve_x(mu, Sigma, lam, r_target):
    n = len(mu)
    x = cp.Variable(n)
    obj = cp.quad_form(x, Sigma) + lam @ x
    prob = cp.Problem(cp.Minimize(obj),
                      [mu @ x >= r_target, cp.sum(x) == 1, x >= 0])
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, warm_start=True)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return np.clip(x.value, 0, None)


# ── z-subproblem (greedy) ─────────────────────────────────────────────────────

def solve_z(lam, K):
    """Select K assets with largest λ_i (minimises -λ'z subject to card ≤ K)."""
    z = np.zeros(len(lam))
    idx = np.argsort(lam)[::-1][:K]
    z[idx] = 1.0
    return z


# ── feasible upper bound ───────────────────────────────────────────────────────

def feasible_portfolio(x, z, mu, Sigma, r_target):
    """Project x onto support(z): re-solve QP restricted to selected assets."""
    idx = np.where(z > 0.5)[0]
    if len(idx) == 0:
        return None, np.inf
    n_full = len(x)
    xk = cp.Variable(len(idx))
    mu_k = mu[idx]; Sigma_k = Sigma[np.ix_(idx, idx)]
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(xk, Sigma_k)),
        [mu_k @ xk >= r_target, cp.sum(xk) == 1, xk >= 0]
    )
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, warm_start=True)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, np.inf
    w = np.zeros(n_full)
    w[idx] = np.clip(xk.value, 0, None)
    obj = float(w @ Sigma @ w)
    return w, obj


# ── subgradient loop ──────────────────────────────────────────────────────────

def lagrangian_relaxation(mu, Sigma, r_target, K,
                          max_iter=300, tol=1e-5):
    n   = len(mu)
    lam = np.zeros(n)           # multipliers ≥ 0
    UB  = np.inf
    LB  = -np.inf
    best_w = None
    history = []

    for t in range(1, max_iter + 1):
        # dual lower bound
        x = solve_x(mu, Sigma, lam, r_target)
        if x is None:
            break
        z = solve_z(lam, K)

        L_val = float(x @ Sigma @ x) + float(lam @ (x - z))
        LB = max(LB, L_val)

        # primal upper bound
        w_feas, ub_val = feasible_portfolio(x, z, mu, Sigma, r_target)
        if ub_val < UB:
            UB = ub_val
            best_w = w_feas

        gap = (UB - LB) / (abs(UB) + 1e-12)
        history.append({"iter": t, "LB": LB, "UB": UB, "gap": gap})

        if gap < tol:
            print(f"    converged at iter {t}  gap={gap:.2e}")
            break

        # Polyak step
        subgrad = x - z
        sg_norm_sq = float(subgrad @ subgrad)
        if sg_norm_sq < 1e-14:
            break
        alpha = (UB - LB) / sg_norm_sq
        lam   = np.maximum(0, lam + alpha * subgrad)

    return best_w, UB, LB, pd.DataFrame(history)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_style()
    print("=" * 55)
    print("Lagrangian Relaxation — Cardinality Portfolio")
    print("=" * 55)

    ret = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    ret = ret.iloc[:, :100]
    mu    = ret.mean().values * 252
    Sigma = ret.cov().values  * 252
    r_target = float(np.percentile(mu, 60))

    K_values = [10, 20, 30]
    all_results = {}

    fig, axes = plt.subplots(1, len(K_values), figsize=(14, 4), sharey=False)

    for ax, K in zip(axes, K_values):
        print(f"\n  K = {K}")
        t0 = time.perf_counter()
        w, UB, LB, hist = lagrangian_relaxation(mu, Sigma, r_target, K)
        elapsed = time.perf_counter() - t0

        final_gap = hist["gap"].iloc[-1]
        port_vol  = float(np.sqrt(w @ Sigma @ w)) if w is not None else np.nan
        port_ret  = float(mu @ w)                 if w is not None else np.nan
        sr        = (port_ret - 0.04) / port_vol  if w is not None else np.nan

        print(f"    UB={UB:.6f}  LB={LB:.6f}  gap={final_gap:.4%}  "
              f"μ={port_ret:.2%}  σ={port_vol:.2%}  SR={sr:.2f}  [{elapsed:.1f}s]")

        all_results[K] = {
            "UB": UB, "LB": LB, "gap": final_gap,
            "port_vol": port_vol, "port_ret": port_ret,
            "sharpe": sr, "solve_sec": round(elapsed, 2),
            "n_iter": len(hist),
        }

        ax.plot(hist["iter"], hist["UB"], label="Upper Bound", color="#dc2626", lw=1.8)
        ax.plot(hist["iter"], hist["LB"], label="Lower Bound", color="#2563eb", lw=1.8,
                linestyle="--")
        ax.fill_between(hist["iter"], hist["LB"], hist["UB"],
                        alpha=0.12, color="#6366f1")
        ax.set_title(f"K = {K}  |  gap = {final_gap:.2%}", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective (annualised variance)")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Lagrangian Relaxation Convergence  —  target return = {r_target:.1%}",
        fontsize=11)
    save_fig("fig3_lagrangian_convergence")
    save_json(all_results, OUTPUT_DIR / "lagrangian_results.json")
    print("\n  Done.")
    return all_results


if __name__ == "__main__":
    main()
