"""
Cardinality-Constrained Mean-Variance Portfolio (CCMVP) via MIQP.

Formulation
-----------
    min   x' Σ x
    s.t.  μ'x ≥ r_target
          1'x  = 1
          x_i  ≥ 0           ∀ i
          x_i  ≤ z_i          ∀ i      (big-M = 1 since x_i ≤ 1)
          Σ z_i ≤ K
          z_i ∈ {0,1}

Solver priority: Gurobi (academic licence) → PuLP/CBC fallback.

Outputs
-------
outputs/fig2_cardinality_frontier.png
outputs/cardinality_results.csv
"""
import time
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import DATA_DIR, OUTPUT_DIR, set_style, save_fig

OUTPUT_DIR.mkdir(exist_ok=True)

# ── solver wrapper ─────────────────────────────────────────────────────────────

def solve_ccmvp_gurobi(mu, Sigma, r_target, K):
    import gurobipy as gp
    from gurobipy import GRB
    n = len(mu)
    m = gp.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 120)

    x = m.addMVar(n, lb=0.0, ub=1.0, name="x")
    z = m.addMVar(n, vtype=GRB.BINARY, name="z")

    m.setObjective(x @ Sigma @ x, GRB.MINIMIZE)
    m.addConstr(mu @ x >= r_target, "ret")
    m.addConstr(x.sum() == 1, "budget")
    m.addConstr(x <= z, "link")
    m.addConstr(z.sum() <= K, "card")

    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0

    if m.Status in (2, 9):          # optimal or time-limit with solution
        return x.X, float(m.ObjVal), elapsed
    return None, None, elapsed


def solve_ccmvp_cvxpy_topk(mu, Sigma, r_target, K):
    """
    CVXPY fallback (no Gurobi):
    1. Solve unconstrained MVO QP to get continuous weights.
    2. Keep top-K by weight → binary support z.
    3. Re-solve restricted QP on support(z) to get min-variance allocation.
    """
    n = len(mu)
    t0 = time.perf_counter()

    # Step 1: continuous MVO
    x = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)),
                      [mu @ x >= r_target, cp.sum(x) == 1])
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        return None, None, time.perf_counter() - t0

    w_cont = np.clip(x.value, 0, None)

    # Step 2: select top-K assets
    idx = np.argsort(w_cont)[::-1][:K]

    # Step 3: restricted QP on top-K assets
    xk = cp.Variable(K, nonneg=True)
    mu_k = mu[idx]; Sig_k = Sigma[np.ix_(idx, idx)]
    prob2 = cp.Problem(cp.Minimize(cp.quad_form(xk, Sig_k)),
                       [mu_k @ xk >= r_target, cp.sum(xk) == 1])
    prob2.solve(solver=cp.OSQP, verbose=False)
    elapsed = time.perf_counter() - t0

    if prob2.status in ("optimal", "optimal_inaccurate") and xk.value is not None:
        w = np.zeros(n)
        w[idx] = np.clip(xk.value, 0, None)
        return w, float(w @ Sigma @ w), elapsed
    return None, None, elapsed


def solve_ccmvp(mu, Sigma, r_target, K):
    try:
        return solve_ccmvp_gurobi(mu, Sigma, r_target, K)
    except ImportError:
        print("  [Gurobi not found — using CVXPY top-K fallback]")
        return solve_ccmvp_cvxpy_topk(mu, Sigma, r_target, K)


# ── main experiment ────────────────────────────────────────────────────────────

def main():
    set_style()
    print("=" * 55)
    print("Cardinality-Constrained Portfolio (MIQP)")
    print("=" * 55)

    ret = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    ret = ret.iloc[:, :100]
    mu    = ret.mean().values * 252
    Sigma = ret.cov().values  * 252
    n     = len(mu)

    # target return = 60th-percentile of asset returns (achievable)
    r_target = float(np.percentile(mu, 60))
    K_values = [5, 10, 15, 20, 30, 50]

    records = []
    weights_all = {}

    for K in K_values:
        w, obj, t = solve_ccmvp(mu, Sigma, r_target, K)
        if w is not None:
            port_vol    = np.sqrt(float(w @ Sigma @ w))
            port_ret    = float(mu @ w)
            n_selected  = int((w > 1e-4).sum())
            sr          = (port_ret - 0.04) / port_vol
            records.append({"K": K, "n_selected": n_selected,
                             "return": port_ret, "vol": port_vol,
                             "sharpe": sr, "solve_sec": round(t, 3)})
            weights_all[K] = w
            print(f"  K={K:3d} → n={n_selected:2d}  μ={port_ret:.2%}  "
                  f"σ={port_vol:.2%}  SR={sr:.2f}  [{t:.1f}s]")
        else:
            print(f"  K={K:3d} → infeasible")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "cardinality_results.csv", index=False)

    # ── plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(df["K"], df["vol"] * 100, "o-", color="#2563eb", lw=2)
    axes[0].set(xlabel="Cardinality K", ylabel="Portfolio Volatility (%)",
                title="Volatility vs. K")

    axes[1].plot(df["K"], df["sharpe"], "s-", color="#dc2626", lw=2)
    axes[1].set(xlabel="Cardinality K", ylabel="Sharpe Ratio",
                title="Sharpe Ratio vs. K")

    axes[2].plot(df["K"], df["solve_sec"], "^-", color="#16a34a", lw=2)
    axes[2].set(xlabel="Cardinality K", ylabel="Solve Time (s)",
                title="Computation Time vs. K")

    fig.suptitle(
        f"Cardinality-Constrained MIQP  —  {n} assets, target return={r_target:.1%}",
        fontsize=11, y=1.02)
    save_fig("fig2_cardinality_frontier")

    # ── weight distribution for K=10, K=30 ────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(9, 4))
    for K, color in zip([10, 20, 30], ["#2563eb", "#dc2626", "#16a34a"]):
        if K in weights_all:
            w = weights_all[K]
            sel = np.sort(w[w > 1e-4])[::-1]
            ax.bar(range(len(sel)), sel * 100, alpha=0.7, label=f"K={K}",
                   color=color, width=0.6)
    ax.set(xlabel="Selected Assets (ranked by weight)",
           ylabel="Portfolio Weight (%)",
           title="Weight Distribution by Cardinality Constraint")
    ax.legend()
    save_fig("fig2b_weight_distribution")

    print(f"\n  Results saved → outputs/cardinality_results.csv")
    return df


if __name__ == "__main__":
    main()
